#include "ReductionRealization.h"
#include "../Value/SourceReplay.h"
#include "../Value/ReplayPolicy.h"
#include "ReductionParameters.h"
#include "ReductionValues.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::gpu::reduction {
namespace {

FragmentType eraseFragmentAxis(FragmentType source, unsigned erasedAxis) {
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  for (auto [axis, extent] : llvm::enumerate(source.getShape())) {
    if (axis == erasedAxis)
      continue;
    shape.push_back(extent);
    auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        source.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
  }
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           ArrayAttr::get(source.getContext(), mappings),
                           source.getValidity(), source.getOwner());
}

FragmentType eraseFragmentAxes(FragmentType source,
                               ArrayRef<int64_t> erasedAxes) {
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  for (auto [axis, extent] : llvm::enumerate(source.getShape())) {
    if (llvm::is_contained(erasedAxes, static_cast<int64_t>(axis)))
      continue;
    shape.push_back(extent);
    auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        source.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
  }
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           ArrayAttr::get(source.getContext(), mappings),
                           source.getValidity(), source.getOwner());
}

FailureOr<bool> decomposeFullCoverageMultiAxisReduce(
    ReduceOp reduce, func::FuncOp kernel, unsigned outerAxis) {
  if (failed(proveLaneWiseHelper(reduce.getCombine()))) return false;

  auto firstSource =
      dyn_cast<FragmentType>(reduce.getSources().front().getType());
  if (!firstSource || outerAxis >= firstSource.getShape().size())
    return false;
  auto outerExtent =
      dyn_cast<PhysicalExprAttr>(firstSource.getShape()[outerAxis]);
  FailureOr<ParameterAttr> coverage =
      outerExtent ? fullCoverageParameter(kernel, outerExtent)
                  : FailureOr<ParameterAttr>(failure());
  if (succeeded(coverage)) {
    auto dimension = coverage->getBinding().getDimension();
    if (!dimension || failed(bindFullCoverageDimension(
            kernel, dimension.getInt(), parameterValue(kernel, *coverage))))
      return reduce.emitOpError(
                 "multi-axis full-coverage fragment could not bind its logical dimension"),
             failure();
  } else {
    auto extent = outerExtent ? constantPhysicalExpression(outerExtent) : std::nullopt;
    if (!extent || *extent <= 0)
      return false;
    for (Value source : reduce.getSources()) {
      auto ranges = PhysicalProgramAnalysis(kernel).axisRanges(source, outerAxis);
      if (!ranges.blockers.empty() || !sameFullOrdinalTraversal(ranges.roots) ||
          exceedsRegisterFile(source, kernel))
        return false;
      auto size = constantLogicalRangeCardinality(ranges.roots.front());
      if (!size || *size > *extent)
        return false;
    }
    // A static full-ordinal workset can trade outer reduction width for more
    // free output lanes. Replay legality alone does not justify replacing a
    // deferred full-coverage dimension: its inner traversal already streams
    // the source, while another outer carry would add a tuple combine per tile.
    if (canChunkReductionOuter(reduce, kernel, outerAxis))
      return false;
  }

  SmallVector<int64_t> innerAxes;
  for (int64_t axis : reduce.getAxes())
    if (axis != static_cast<int64_t>(outerAxis))
      innerAxes.push_back(axis);
  if (innerAxes.empty())
    return false;

  SmallVector<Value> identities(
      reduce.getIdentities()
          .begin(),
      reduce.getIdentities()
          .end());
  ValueRange captures = reduce.getCaptures();
  SmallVector<Value> innerIdentities;
  OpBuilder builder(reduce);
  for (auto [component, source] : llvm::enumerate(
           reduce.getSources())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || outerAxis >= fragment.getShape().size())
      return reduce.emitOpError(
                 "multi-axis full-coverage source lost its fragment schema"),
             failure();
    auto inferred = inferCollectiveResultType(
        fragment, innerAxes, dataElementType(identities[component].getType()));
    if (failed(inferred))
      return reduce.emitOpError(
                 "multi-axis inner reduction axes do not match its source schema"),
             failure();
    auto result = dyn_cast<FragmentType>(*inferred);
    if (!result)
      return reduce.emitOpError(
                 "multi-axis inner reduction did not retain its outer axis"),
             failure();
    FailureOr<Value> identity = alignToExecutionSchema(
        builder, reduce.getLoc(), identities[component], result);
    if (failed(identity))
      return reduce.emitOpError(
                 "multi-axis identity cannot be projected over the retained fragment"),
             failure();
    innerIdentities.push_back(*identity);
  }

  auto innerReduce = builder.create<ReduceOp>(reduce.getLoc(),
      reduce.getSources(), innerIdentities, captures, innerAxes);
  if (Attribute origin = reduce->getAttr(originAttr))
    innerReduce->setAttr(originAttr, origin);
  std::string reason;
  if (failed(liftCombineRegion(reduce.getCombine(),
                                      innerReduce.getCombine(),
                                      innerReduce.getResultTypes(), reason)))
    return reduce.emitOpError(
               "multi-axis combine cannot execute over its retained fragment: ")
           << reason;

  unsigned outerResultAxis = 0;
  for (unsigned axis = 0; axis < outerAxis; ++axis)
    if (!llvm::is_contained(innerAxes, static_cast<int64_t>(axis)))
      ++outerResultAxis;
  auto outerReduce = builder.create<ReduceOp>(reduce.getLoc(),
      innerReduce.getResults(), identities, captures,
      ArrayRef<int64_t>{static_cast<int64_t>(outerResultAxis)});
  if (Attribute origin = reduce->getAttr(originAttr))
    outerReduce->setAttr(originAttr, origin);
  IRMapping regionMapping;
  reduce.getCombine().cloneInto(&outerReduce.getCombine(), regionMapping);

  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), outerReduce.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  eraseDeadPhysicalValues(kernel);
  return true;
}

} // namespace

LogicalResult decomposeMultiAxisReduce(ReduceOp reduce, func::FuncOp kernel) {
  if (!reduce->getBlock() || reduce.getAxes().size() <= 1)
    return success();
  if (reduce.getSources().size() == 0)
    return reduce.emitOpError(
        "multi-axis reduction decomposition requires at least one source");

  unsigned outerAxis = static_cast<unsigned>(reduce.getAxes().front());
  FailureOr<bool> fullCoverage = shouldTileReductionSources(reduce, kernel)
      ? FailureOr<bool>(false)
      : decomposeFullCoverageMultiAxisReduce(reduce, kernel, outerAxis);
  if (failed(fullCoverage))
    return failure();
  if (*fullCoverage)
    return success();
  SmallVector<SourcePlan> plans;
  SmallVector<SmallVector<RootAccess>> accesses;
  for (Value source : reduce.getSources()) {
    FailureOr<SourcePlan> plan = analyzeSource(source, outerAxis);
    if (failed(plan)) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      FailureOr<AxisMapAttr> mapping =
          fragment ? queryAxisMap(fragment, outerAxis)
                   : FailureOr<AxisMapAttr>(failure());
      if (!fragment || failed(mapping) ||
          !isReplayableWithoutLoad(source, sourceAxisIdentity(*mapping))) {
        auto diagnostic = reduce.emitOpError()
            << "multi-axis reduction outer axis is neither load-rooted nor a replayable pure source; source type="
            << source.getType() << ", producer="
            << (source.getDefiningOp()
                    ? source.getDefiningOp()->getName().getStringRef()
                    : StringRef("block argument"));
        auto ranges = PhysicalProgramAnalysis(kernel).axisRanges(source, outerAxis);
        diagnostic << "; axis=" << outerAxis;
        for (MakeRangeOp range : ranges.roots)
          diagnostic << "; range=" << range.getResult();
        return failure();
      }
      plan = SourcePlan{source, sourceAxisIdentity(*mapping), outerAxis, {}, {},
                        {}};
    }
    SmallVector<RootAccess> roots;
    for (LoadOp load : plan->roots) {
      FailureOr<std::optional<RootAccess>> access =
          analyzeRoot(load, plan->ranges, plan->reductionAxis);
      if (failed(access) || !*access)
        return reduce.emitOpError(
            "multi-axis reduction outer axis has no unit-step load coordinate");
      roots.push_back(**access);
    }
    plans.push_back(*plan);
    accesses.push_back(std::move(roots));
  }

  std::optional<RootAccess> master;
  for (const auto &component : accesses)
    if (!component.empty()) {
      master = component.front();
      break;
    }
  if (!master)
    return reduce.emitOpError(
        "multi-axis reduction has no load-rooted traversal authority");
  auto sharesOuterTraversal = [&](MakeRangeOp range) {
    return PhysicalProgramAnalysis(kernel)
        .lockstepRanges({range, master->range}).isExact();
  };
  for (const auto &component : accesses)
    for (RootAccess access : component) {
      if (!sharesOuterTraversal(access.range))
        return reduce.emitOpError(
            "multi-axis reduction components do not share one exact outer traversal");
    }
  for (const SourcePlan &plan : plans)
    if (plan.reductionRange && !sharesOuterTraversal(plan.reductionRange))
      return reduce.emitOpError(
          "multi-axis reduction sources do not share one exact outer traversal");

  for (const auto &component : accesses)
    for (RootAccess access : component)
      if (failed(prepareSourceReplay(access.load.getResult(), access.fragmentAxis,
                                     reduce, PhysicalReplayScope::ValueGraph)))
        return reduce.emitOpError("reduction could not preserve its source snapshot");

  SmallVector<int64_t> retainedInnerAxes;
  SmallVector<int64_t> innerAxes;
  for (int64_t axis : reduce.getAxes()) {
    if (axis == static_cast<int64_t>(outerAxis))
      continue;
    retainedInnerAxes.push_back(axis);
    innerAxes.push_back(axis > static_cast<int64_t>(outerAxis) ? axis - 1
                                                               : axis);
  }
  if (innerAxes.empty())
    return reduce.emitOpError(
        "multi-axis reduction decomposition lost every inner axis");

  SmallVector<Value> identities(
      reduce.getIdentities()
          .begin(),
      reduce.getIdentities()
          .end());
  ValueRange captures = reduce.getCaptures();
  PhysicalExprAttr unitExtent =
      expression(reduce.getContext(), PhysicalExprKind::Constant, 1);
  bool outerHasSubregion = llvm::any_of(
      accesses, [](ArrayRef<RootAccess> component) {
        return llvm::any_of(component, [](RootAccess access) {
          return access.range->hasAttr(sourceSubregionAttr);
        });
      });
  for (const SourcePlan &plan : plans)
    outerHasSubregion |= llvm::any_of(plan.ranges, [&](MakeRangeOp range) {
      return sourceAxisIdentity(range) == sourceAxisIdentity(master->range) &&
             range->hasAttr(sourceSubregionAttr);
    });
  bool blockOuterAxis = !outerHasSubregion &&
                        succeeded(proveLaneWiseHelper(reduce.getCombine()));
  ParameterAttr outerChunk;
  PhysicalExprAttr outerSliceExtent = unitExtent;
  if (blockOuterAxis) {
    std::string name =
        ("REDUCE_CHUNK_" + Twine(master->range.getSourceId()) + "_A" +
         Twine(master->range.getSourceAxis()) +
         (master->range.getDerived() ? "_DERIVED" : ""))
            .str();
    SmallVector<int64_t> candidates{1, 2, 4, 8, 16, 32, 64, 128,
                                    256, 512, 1024, 2048, 4096};
    if (auto count = constantLogicalRangeCardinality(master->range)) {
      uint64_t padded = llvm::PowerOf2Ceil(
          static_cast<uint64_t>(std::max<int64_t>(*count, 1)));
      llvm::erase_if(candidates, [&](int64_t value) {
        return static_cast<uint64_t>(value) > padded;
      });
      name += ("_E" + Twine(padded)).str();
    }
    auto firstSource =
        cast<FragmentType>(reduce.getSources().front().getType());
    if (queryFragmentAxes(firstSource, plans.front().sourceIdentity).size() > 1)
      name += ("_F" + Twine(outerAxis)).str();
    auto selected = selectReductionChunk(
        kernel, sourceAxisIdentity(master->range), master->range,
        ParameterRole::ReductionOuter,
        firstSource.getElementType().getIntOrFloatBitWidth(), name, candidates);
    if (failed(selected))
      return reduce.emitOpError(
          "multi-axis outer reduction has no physical chunk parameter");
    outerChunk = *selected;
    auto bounded = boundedTraversalChunk(outerChunk, master->range);
    if (failed(bounded))
      return reduce.emitOpError("multi-axis outer chunk has no bounded source capacity");
    outerSliceExtent = *bounded;
  }
  OpBuilder builder(reduce);
  Location location = reduce.getLoc();
  Value outerChunkValue;
  if (blockOuterAxis)
    outerChunkValue = outerSliceExtent.getKind() == PhysicalExprKind::Parameter
        ? Value(materializeParameter(builder, location, outerSliceExtent.getParameterReference()))
        : Value(builder.create<PhysicalExprOp>(location, builder.getIndexType(), outerSliceExtent));
  SmallVector<FragmentType> accumulatorTypes;
  if (blockOuterAxis)
    for (const SourcePlan &plan : plans)
      accumulatorTypes.push_back(eraseFragmentAxes(
          replaceExtent(cast<FragmentType>(plan.source.getType()), outerAxis,
                        outerSliceExtent),
          retainedInnerAxes));
  Region vectorCombine;
  bool vectorAccumulation =
      blockOuterAxis &&
      prepareVectorAccumulation(reduce, accumulatorTypes, vectorCombine);
  // Retain outer-axis partials across chunks and reduce those lanes once.
  SmallVector<Value> loopInitials(identities);
  if (vectorAccumulation) {
    loopInitials.clear();
    for (auto [identity, type] : llvm::zip(identities, accumulatorTypes))
      loopInitials.push_back(builder.create<SplatOp>(
          location, type, *scalarSource(identity)));
  }
  unsigned outerResultAxis = 0;
  for (unsigned axis = 0; axis < outerAxis; ++axis)
    if (!llvm::is_contained(retainedInnerAxes, static_cast<int64_t>(axis)))
      ++outerResultAxis;
  auto reduceOuterAxis = [&](OpBuilder &at, Location loc,
                             ValueRange sources) {
    auto outerReduce = at.create<ReduceOp>(loc, sources,
        identities, captures, ArrayRef<int64_t>{static_cast<int64_t>(outerResultAxis)});
    if (Attribute origin = reduce->getAttr(originAttr))
      outerReduce->setAttr(originAttr, origin);
    IRMapping mapping;
    reduce.getCombine().cloneInto(&outerReduce.getCombine(), mapping);
    return SmallVector<Value>(outerReduce.getResults().begin(),
                              outerReduce.getResults().end());
  };
  bool bodyFailed = false;
  std::string failureReason;
  ReplayPolicy reuse(kernel, reduce.getSources(), {reduce.getOperation()});
  Value outerLoopStep =
      blockOuterAxis ? outerChunkValue : master->range.getStep();
  auto loop = builder.create<scf::ForOp>(
      location, master->range.getLogicalStart(),
      master->range.getLogicalStop(), outerLoopStep, loopInitials,
      [](OpBuilder &, Location, Value, ValueRange) {});
  OpBuilder::atBlockEnd(loop.getBody()).create<scf::YieldOp>(location, loop.getRegionIterArgs());
  // Replay analyses require the new values to belong to the current kernel.
  // A ForOp build callback runs before the loop is attached to that kernel.
  auto buildBody = [&](OpBuilder &nested, Location nestedLocation, Value coordinate,
          ValueRange carries) {
        SmallVector<Value> innerSources;
        for (auto [component, plan] : llvm::enumerate(plans)) {
          IRMapping mapping;
          auto mapOuterRange = [&](MakeRangeOp range) -> FailureOr<Value> {
            if (Value mapped = mapping.lookupOrNull(range.getResult()))
              return mapped;
            auto rangeType = dyn_cast<FragmentType>(range.getResult().getType());
            if (!rangeType || rangeType.getShape().size() != 1)
              return failure();
            FragmentType blockedType =
                replaceExtent(rangeType, 0, outerSliceExtent);
            if (!blockOuterAxis) {
              auto slice = nested.create<BroadcastOp>(
                  nestedLocation, blockedType, coordinate);
              mapping.map(range.getResult(), slice.getResult());
              return slice.getResult();
            }
            auto blocked = nested.create<MakeRangeOp>(
                nestedLocation, blockedType, coordinate,
                outerChunkValue, range.getStep(),
                range.getLogicalStart(), range.getLogicalStop(),
                range.getSourceId(), range.getSourceAxis(), range.getDerived());
            inheritRangeAuthority(blocked, range);
            mapping.map(range.getResult(), blocked.getResult());
            return blocked.getResult();
          };
          // Tensor users retain a one-coordinate fragment. Its scalar loop
          // slice must not become a new range for later full-domain blocking.
          for (MakeRangeOp range : plan.ranges)
            if (failed(mapOuterRange(range))) {
              bodyFailed = true;
              failureReason =
                  "outer reduction range could not be blocked";
              return;
            }
          auto sourceType = cast<FragmentType>(plan.source.getType());
          auto sourceAxis = cast<AxisMapAttr>(sourceType.getAxisMaps()[plan.reductionAxis]);
          auto sourceCoordinate = mapOuterRange(plan.ranges.empty()
              ? master->range : plan.ranges.front());
          if (failed(sourceCoordinate) ||
              failed(bindSourceReplay(nested, nestedLocation, plan.source,
                  plan.sourceIdentity, sourceAxis.getDimensionId(), plan.reductionAxis,
                  outerSliceExtent, *sourceCoordinate, reduce,
                  PhysicalReplayScope::ValueGraph, {}, mapping, &reuse))) {
            bodyFailed = true;
            failureReason = "could not bind the current outer source snapshot";
            return;
          }
          auto remaining = PhysicalProgramAnalysis(kernel).replayAt(
              plan.source, plan.sourceIdentity, PhysicalReplayScope::ValueGraph,
              true, reduce, mapping, sourceAxis.getDimensionId());
          if (!remaining.isReplayable()) {
            bodyFailed = true;
            failureReason = "bound outer source has no complete producer replay proof";
            return;
          }
          for (RootAccess access : accesses[component]) {
            if (mapping.lookupOrNull(access.load.getResult()) ||
                !llvm::is_contained(remaining.accesses, access.load.getOperation())) continue;
            if (failed(mapOuterRange(access.range))) {
              bodyFailed = true;
              failureReason =
                  "outer reduction load range could not be blocked";
              return;
            }
            LoadOp load = access.load;
            auto sourceType = cast<FragmentType>(load.getResult().getType());
            auto accessAxis =
                cast<AxisMapAttr>(sourceType.getAxisMaps()[access.fragmentAxis]);
            PhysicalSourceAxis accessSource = sourceAxisIdentity(accessAxis);
            FragmentType slicedType = replaceExtent(
                sourceType, access.fragmentAxis, outerSliceExtent);
            SmallVector<Value> coordinates(load.getCoordinates());
            // Coordinates, validity and fill can hold separate SSA occurrences
            // of the same range. Bind all of them before replaying the access.
            SmallVector<Value> accessValues(load.getCoordinates());
            if (load.getValid())
              accessValues.push_back(load.getValid());
            if (load.getFill())
              accessValues.push_back(load.getFill());
            SmallVector<MakeRangeOp> companionRanges;
            PhysicalProgramAnalysis rangeAnalysis(kernel);
            for (Value value : accessValues)
              for (MakeRangeOp range : rangeAnalysis.sourceRanges(value).roots)
                if (!llvm::is_contained(companionRanges, range))
                  companionRanges.push_back(range);
            for (MakeRangeOp range : companionRanges) {
              if (mapping.lookupOrNull(range.getResult()))
                continue;
              auto dimension = queryRangeDimension(range);
              auto accessDimension = queryRangeDimension(access.range);
              if (sameLogicalRange(range, access.range) &&
                  succeeded(dimension) && succeeded(accessDimension) &&
                  *dimension == *accessDimension &&
                  samePhysicalScalarExpression(range.getStart(),
                                               access.range.getStart())) {
                if (failed(mapOuterRange(range))) {
                  bodyFailed = true;
                  failureReason = "outer access companion range could not be blocked";
                  return;
                }
                continue;
              }
              auto rangeType = cast<FragmentType>(range.getResult().getType());
              auto clone = nested.create<MakeRangeOp>(
                  nestedLocation, rangeType, range.getStart(),
                  range.getExtent(), range.getStep(), range.getLogicalStart(),
                  range.getLogicalStop(), range.getSourceId(),
                  range.getSourceAxis(), range.getDerived());
              inheritRangeAuthority(clone, range);
              mapping.map(range.getResult(), clone.getResult());
              mapping.map(clone.getResult(), clone.getResult());
            }
            FailureOr<Value> reducedCoordinate = materializeReplayedValue(
                nested, nestedLocation,
                load.getCoordinates()[access.coordinateIndex],
                accessSource, outerSliceExtent, mapping, reduce);
            if (failed(reducedCoordinate)) {
              bodyFailed = true;
              failureReason =
                  "reduced source coordinate could not be cloned into the outer loop";
              return;
            }
            coordinates[access.coordinateIndex] = *reducedCoordinate;
            using ReplayAxis =
                std::pair<PhysicalSourceAxis, PhysicalExprAttr>;
            SmallVector<ReplayAxis> replayAxes{
                {accessSource, outerSliceExtent}};
            for (auto [coordinateIndex, original] :
                 llvm::enumerate(load.getCoordinates())) {
              if (coordinateIndex == access.coordinateIndex)
                continue;
              MakeRangeOp range = sourceRange(original);
              PhysicalSourceAxis coordinateSource = accessSource;
              PhysicalExprAttr coordinateExtent = outerSliceExtent;
              ReplayMaterializationOptions coordinateOptions;
              auto coordinateType = dyn_cast<FragmentType>(original.getType());
              BroadcastProjection relation =
                  coordinateType ? queryAxisProjection(coordinateType, sourceType)
                                 : BroadcastProjection{};
              if (relation.isExact() &&
                  relation.targetToSource[access.fragmentAxis]) {
                unsigned coordinateAxis =
                    *relation.targetToSource[access.fragmentAxis];
                coordinateOptions.fragmentAxis = coordinateAxis;
                coordinateSource = sourceAxisIdentity(cast<AxisMapAttr>(
                    coordinateType.getAxisMaps()[coordinateAxis]));
              } else if (range) {
                auto rangeType = cast<FragmentType>(range.getResult().getType());
                coordinateSource = sourceAxisIdentity(range);
                coordinateExtent = cast<PhysicalExprAttr>(rangeType.getShape()[0]);
                replayAxes.emplace_back(coordinateSource, coordinateExtent);
              }
              FailureOr<Value> replayedCoordinate = materializeReplayedValue(
                  nested, nestedLocation, original, coordinateSource,
                  coordinateExtent, mapping, reduce, coordinateOptions);
              if (failed(replayedCoordinate)) {
                bodyFailed = true;
                failureReason =
                    "non-reduced source coordinate could not be cloned into the outer loop";
                return;
              }
              coordinates[coordinateIndex] = *replayedCoordinate;
            }
            Value valid;
            if (load.getValid()) {
              valid = load.getValid();
              // Reuse the coordinate SSA in its bounds predicates and fill.
              for (auto [source, extent] : replayAxes) {
                FailureOr<Value> replayed = materializeReplayedValue(
                    nested, nestedLocation, valid, source, extent,
                    mapping, reduce);
                if (failed(replayed)) {
                  bodyFailed = true;
                  failureReason =
                      "source validity could not be cloned into the outer loop";
                  return;
                }
                valid = *replayed;
              }
            }
            Value fill;
            if (load.getFill()) {
              fill = load.getFill();
              for (auto [source, extent] : replayAxes) {
                FailureOr<Value> replayed = materializeReplayedValue(
                    nested, nestedLocation, fill, source, extent,
                    mapping, reduce);
                if (failed(replayed)) {
                  bodyFailed = true;
                  failureReason =
                      "source fill could not be cloned into the outer loop";
                  return;
                }
                fill = *replayed;
              }
            }
            auto slicedLoad = materializeSourceRead(
                nested, nestedLocation, access.load, access.fragmentAxis, slicedType, coordinates,
                valid, fill, mapping.lookupOrNull(access.range.getResult()),
                reduce);
            if (failed(slicedLoad)) {
              bodyFailed = true;
              failureReason = "outer reduction could not preserve its source read";
              return;
            }
            mapping.map(load.getResult(), *slicedLoad);
          }
          ReplayMaterializationOptions replayOptions;
          replayOptions.fragmentAxis = plan.reductionAxis;
          FailureOr<Value> replayed = materializeReplayedValue(
              nested, nestedLocation, plan.source, plan.sourceIdentity,
              outerSliceExtent, mapping, reduce, replayOptions);
          if (failed(replayed)) {
            bodyFailed = true;
            failureReason = "outer-axis source graph could not be sliced";
            return;
          }
          auto sliced = dyn_cast<FragmentType>((*replayed).getType());
          if (!sliced || outerAxis >= sliced.getShape().size()) {
            bodyFailed = true;
            failureReason = "outer-axis source lost its fragment schema";
            return;
          }
          if (blockOuterAxis) {
            auto outerMap =
                cast<AxisMapAttr>(sliced.getAxisMaps()[outerAxis]);
            auto rangeType = FragmentType::get(
                reduce.getContext(), nested.getIndexType(),
                nested.getArrayAttr({outerSliceExtent}),
                nested.getArrayAttr({AxisMapAttr::get(
                    reduce.getContext(), outerMap.getSourceId(),
                    outerMap.getSourceAxis(), outerMap.getDimensionId(),
                    /*fragmentAxis=*/0, outerMap.getDerived())}),
                sliced.getValidity(), sliced.getOwner());
            Value physicalExtent = outerChunkValue;
            auto outerRange = nested.create<MakeRangeOp>(
                nestedLocation, rangeType, coordinate, physicalExtent,
                master->range.getStep(), master->range.getLogicalStart(),
                master->range.getLogicalStop(), outerMap.getSourceId(),
                outerMap.getSourceAxis(), outerMap.getDerived());
            Value logicalEnd = nested.create<BroadcastOp>(
                nestedLocation, rangeType, master->range.getLogicalStop());
            auto predicateType = withElementType(rangeType, nested.getI1Type());
            Value active = nested.create<CompareOp>(
                nestedLocation, predicateType, outerRange.getResult(),
                logicalEnd, ComparePredicate::Lt);
            FailureOr<Value> alignedActive = projectPredicateToFragmentAxis(
                nested, nestedLocation, active, sliced, outerAxis);
            FailureOr<Value> alignedIdentity = alignToExecutionSchema(
                nested, nestedLocation, identities[component], sliced);
            if (failed(alignedActive) || failed(alignedIdentity)) {
              bodyFailed = true;
              failureReason =
                  "outer reduction tail could not be aligned to its source";
              return;
            }
            innerSources.push_back(nested.create<SelectOp>(
                nestedLocation, sliced, *alignedActive, *replayed,
                *alignedIdentity));
            continue;
          }
          FragmentType squeezed = eraseFragmentAxis(sliced, outerAxis);
          SmallVector<Attribute> reassociation;
          unsigned resultAxis = 0;
          for (unsigned sourceAxis = 0;
               sourceAxis < sliced.getShape().size(); ++sourceAxis) {
            SmallVector<int64_t> resultAxes;
            if (sourceAxis != outerAxis)
              resultAxes.push_back(resultAxis++);
            reassociation.push_back(ReshapeGroupAttr::get(
                reduce.getContext(),
                DenseI64ArrayAttr::get(
                    reduce.getContext(),
                    {static_cast<int64_t>(sourceAxis)}),
                DenseI64ArrayAttr::get(reduce.getContext(), resultAxes)));
          }
          innerSources.push_back(nested.create<ReshapeOp>(
              nestedLocation, squeezed, *replayed,
              ArrayAttr::get(reduce.getContext(), reassociation)));
        }
        if (bodyFailed)
          return;

        SmallVector<Value> innerIdentities(identities);
        ArrayRef<int64_t> selectedInnerAxes =
            blockOuterAxis ? retainedInnerAxes : innerAxes;
        if (blockOuterAxis)
          innerIdentities.clear();
        for (auto [component, source] : llvm::enumerate(innerSources)) {
          auto inferred = inferCollectiveResultType(source.getType(),
              selectedInnerAxes, dataElementType(identities[component].getType()));
          if (failed(inferred)) {
            bodyFailed = true;
            failureReason = "replayed inner reduction axes do not match its source schema";
            return;
          }
          if (!blockOuterAxis)
            continue;
          auto result = dyn_cast<FragmentType>(*inferred);
          if (!result) {
            bodyFailed = true;
            failureReason = "outer-block inner reduction did not retain its fragment axis";
            return;
          }
          FailureOr<Value> identity = alignToExecutionSchema(
              nested, nestedLocation, identities[component], result);
          if (failed(identity)) {
            bodyFailed = true;
            failureReason =
                "outer-block identity could not retain its fragment axis";
            return;
          }
          innerIdentities.push_back(*identity);
        }

        auto innerReduce = nested.create<ReduceOp>(nestedLocation, innerSources,
            innerIdentities, captures, selectedInnerAxes);
        if (Attribute origin = reduce->getAttr(originAttr))
          innerReduce->setAttr(originAttr, origin);
        if (blockOuterAxis) {
          if (failed(liftCombineRegion(
                  reduce.getCombine(), innerReduce.getCombine(),
                  innerReduce.getResultTypes(), failureReason))) {
            bodyFailed = true;
            return;
          }
        } else {
          IRMapping regionMapping;
          reduce.getCombine().cloneInto(&innerReduce.getCombine(),
                                        regionMapping);
        }

        SmallVector<Value> summaries(innerReduce.getResults().begin(),
                                     innerReduce.getResults().end());
        if (blockOuterAxis && !vectorAccumulation)
          summaries = reduceOuterAxis(nested, nestedLocation, summaries);

        SmallVector<Value> combineArguments(carries.begin(), carries.end());
        combineArguments.append(summaries.begin(), summaries.end());
        combineArguments.append(captures.begin(), captures.end());
        FailureOr<SmallVector<Value>> combined = inlinePureRegion(
            nested, vectorAccumulation ? vectorCombine : reduce.getCombine(),
            combineArguments, failureReason);
        if (failed(combined)) {
          bodyFailed = true;
          return;
        }
        nested.create<scf::YieldOp>(nestedLocation, *combined);
      };
  Operation *placeholder = loop.getBody()->getTerminator();
  OpBuilder bodyBuilder(placeholder);
  buildBody(bodyBuilder, location, loop.getInductionVar(), loop.getRegionIterArgs());
  if (bodyFailed) {
    loop.erase();
    return reduce.emitOpError(
               "multi-axis reduction decomposition failed: ")
           << failureReason;
  }
  placeholder->erase();
  if (Attribute origin = reduce->getAttr(originAttr))
    loop->setAttr(originAttr, origin);
  loop->setAttr(reductionSourcesAttr, reductionSources(reduce));
  SmallVector<Value> realizedResults(loop.getResults().begin(),
                                     loop.getResults().end());
  if (vectorAccumulation)
    realizedResults = reduceOuterAxis(builder, location, realizedResults);
  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), realizedResults))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace intent::gpu::reduction
