#include "ReductionRealization.h"
#include "../Value/SourceReplay.h"
#include "../Value/ReplayPolicy.h"
#include "ReductionParameters.h"
#include "ReductionValues.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"

#include <functional>

using namespace mlir;

namespace intent::gpu::reduction {

LogicalResult realizeRuntimeReduce(ReduceOp reduce,
                                   ArrayRef<SourcePlan> sourcePlans,
                                   func::FuncOp kernel,
                                   bool tileProducerFreeAxis,
                                   ReductionCarryForm carryForm) {
  if (sourcePlans.empty())
    return reduce.emitOpError("runtime reduction has no physical sources");
  SmallVector<SmallVector<RootAccess>> accesses;
  SmallVector<MakeRangeOp> traversalRanges;
  for (const SourcePlan &plan : sourcePlans) {
    auto source = cast<FragmentType>(plan.source.getType());
    for (auto [axis, extent] : llvm::enumerate(source.getShape()))
      if (axis != plan.reductionAxis &&
          !isCompileTimeExtent(cast<PhysicalExprAttr>(extent))) {
        auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
        return reduce.emitOpError(
                   "runtime reduction free axes must be physicalized before chunking")
               << "; free_axis=" << axis << ", reduction_axis="
               << plan.reductionAxis << ", extent=" << extent
               << ", source_id=" << mapping.getSourceId()
               << ", source_axis=" << mapping.getSourceAxis()
               << ", dimension=" << mapping.getDimensionId()
               << ", source=" << source;
      }
    SmallVector<RootAccess> roots;
    for (LoadOp load : plan.roots) {
      FailureOr<std::optional<RootAccess>> access =
          analyzeRoot(load, plan.ranges, plan.reductionAxis);
      if (failed(access) || !*access)
        return reduce.emitOpError(
            "load-rooted producer has no unit-step reduction coordinate");
      roots.push_back(**access);
    }
    SmallVector<MakeRangeOp> componentRanges;
    for (RootAccess access : roots)
      if (!llvm::is_contained(componentRanges, access.range))
        componentRanges.push_back(access.range);
    for (MakeRangeOp range : plan.ranges)
      if (!llvm::is_contained(componentRanges, range))
        componentRanges.push_back(range);
    PhysicalLockstepTraversalFact traversal =
        PhysicalProgramAnalysis(kernel).lockstepRanges(componentRanges);
    if (!traversal.isExact())
      return reduce.emitOpError(
          "one reduction component has no exact lockstep range authority");
    accesses.push_back(std::move(roots));
    traversalRanges.push_back(traversal.authority);
  }
  for (const auto &component : accesses)
    for (RootAccess access : component)
      if (failed(prepareSourceReplay(access.load.getResult(), access.fragmentAxis,
                                     reduce, PhysicalReplayScope::ValueGraph)))
        return reduce.emitOpError("reduction could not preserve its source snapshot");
  for (Type result : reduce.getResultTypes())
    if (auto fragment = dyn_cast<FragmentType>(result))
      if (llvm::any_of(fragment.getShape(), [](Attribute extent) {
            return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
          }))
        return reduce.emitOpError(
            "runtime reduction result still has an unphysicalized free axis");

  MakeRangeOp firstRange = traversalRanges.front();
  Value firstEnd = firstRange.getLogicalStop();
  // Components are paired by the reduce axes, not by allocation provenance.
  // Keep each source identity while proving their actual traversals coincide.
  if (!PhysicalProgramAnalysis(kernel).lockstepRanges(traversalRanges).isExact()) {
    auto diagnostic = reduce.emitOpError(
        "runtime reduction components require one lockstep logical range");
    for (MakeRangeOp range : traversalRanges)
      diagnostic << "; range=" << range.getResult()
                 << "; logical_stop=" << range.getLogicalStop();
    return failure();
  }
  for (MakeRangeOp range : traversalRanges) {
    Value end = range.getLogicalStop();
    if (!samePhysicalScalarExpression(firstEnd, end))
      return reduce.emitOpError(
          "runtime reduction components require one lockstep logical range");
  }
  auto selectedChunk = selectReductionTraversalChunk(
      reduce, sourcePlans.front().source, sourcePlans.front().reductionAxis,
      firstRange, tileProducerFreeAxis);
  if (failed(selectedChunk)) return failure();
  ParameterAttr chunk = *selectedChunk;
  if (!chunk)
    return reduce.emitOpError(
               "reduction blocking has no unique physical parameter relation")
           << "; source=" << sourcePlans.front().source.getType()
           << "; axis=" << sourcePlans.front().reductionAxis;
  auto boundedChunk = boundedTraversalChunk(chunk, firstRange);
  if (failed(boundedChunk)) return failure();
  PhysicalExprAttr chunkExtent = *boundedChunk;

  SmallVector<FragmentType> blockedSourceTypes;
  for (const SourcePlan &plan : sourcePlans) {
    auto originalSource = cast<FragmentType>(plan.source.getType());
    blockedSourceTypes.push_back(
        replaceExtent(originalSource, plan.reductionAxis, chunkExtent));
  }

  OpBuilder builder(reduce);
  Location location = reduce.getLoc();
  Value chunkSize = chunkExtent.getKind() == PhysicalExprKind::Parameter
      ? materializeParameter(builder, location, chunk.getReference()).getResult()
      : builder.create<PhysicalExprOp>(location, builder.getIndexType(), chunkExtent).getResult();
  Value stop = firstEnd;
  SmallVector<Value> identities(
      reduce.getIdentities()
          .begin(),
      reduce.getIdentities()
          .end());
  ValueRange captures = reduce.getCaptures();

  Region vectorCombine;
  bool vectorAccumulation =
      carryForm == ReductionCarryForm::Automatic &&
      prepareVectorAccumulation(reduce, blockedSourceTypes, vectorCombine);

  SmallVector<Value> loopInitials(identities);
  if (vectorAccumulation) {
    loopInitials.clear();
    for (auto [identity, blockedType] :
         llvm::zip(identities, blockedSourceTypes)) {
      FailureOr<Value> scalar = scalarSource(identity);
      if (failed(scalar))
        return reduce.emitOpError(
            "vector reduction accumulation lost its scalar identity");
      Value initial = builder.create<SplatOp>(location, blockedType, *scalar);
      loopInitials.push_back(initial);
    }
  }

  bool bodyFailed = false;
  std::string bodyFailure = "unknown producer replay failure";
  ReplayPolicy reuse(kernel, reduce.getSources(), {reduce.getOperation()});
  auto loop = builder.create<scf::ForOp>(
      location, firstRange.getStart(), stop, chunkSize, loopInitials,
      [](OpBuilder &, Location, Value, ValueRange) {});
  auto buildBody = [&](OpBuilder &nested, Location nestedLocation, Value chunkStart,
          ValueRange carries) {
        auto masterType = cast<FragmentType>(firstRange.getResult().getType());
        SmallVector<Attribute> masterShape(masterType.getShape().begin(),
                                           masterType.getShape().end());
        masterShape[0] = chunkExtent;
        auto blockedMaster = FragmentType::get(
            reduce.getContext(), masterType.getElementType(),
            ArrayAttr::get(reduce.getContext(), masterShape),
            masterType.getAxisMaps(), 2, masterType.getOwner());
        Value masterCoordinate = nested.create<MakeRangeOp>(
            nestedLocation, blockedMaster, chunkStart, chunkSize,
            firstRange.getStep(), firstRange.getLogicalStart(),
            firstRange.getLogicalStop(), firstRange.getSourceId(),
            firstRange.getSourceAxis(), firstRange.getDerived());
        inheritRangeAuthority(masterCoordinate.getDefiningOp(), firstRange);
        Value masterEnd = nested.create<BroadcastOp>(nestedLocation,
                                                     blockedMaster, stop);
        auto masterPredicate = FragmentType::get(
            reduce.getContext(), nested.getI1Type(), blockedMaster.getShape(),
            blockedMaster.getAxisMaps(), 2, blockedMaster.getOwner());
        Value sharedTail = nested.create<CompareOp>(
            nestedLocation, masterPredicate, masterCoordinate, masterEnd,
            ComparePredicate::Lt);
        SmallVector<Value> blockedSources;
        for (auto [component, plan] : llvm::enumerate(sourcePlans)) {
          FragmentType blockedSource = blockedSourceTypes[component];
          auto blockedPredicate = FragmentType::get(
              reduce.getContext(), nested.getI1Type(), blockedSource.getShape(),
              blockedSource.getAxisMaps(), blockedSource.getValidity(),
              blockedSource.getOwner());
          IRMapping mapping;
          ReplayMaterializationOptions replayOptions;
          replayOptions.traversalRanges = plan.ranges;
          Value sourceTail;
          for (MakeRangeOp range : plan.ranges) {
            if (mapping.lookupOrNull(range.getResult()))
              continue;
            auto originalCoordinate = range.getResult().getType();
            SmallVector<Attribute> coordinateShape(
                originalCoordinate.getShape().begin(),
                originalCoordinate.getShape().end());
            coordinateShape[0] = chunkExtent;
            auto blockedCoordinate = FragmentType::get(
                reduce.getContext(), originalCoordinate.getElementType(),
                ArrayAttr::get(reduce.getContext(), coordinateShape),
                originalCoordinate.getAxisMaps(), originalCoordinate.getValidity(),
                originalCoordinate.getOwner());
            auto coordinate = nested.create<MakeRangeOp>(
                nestedLocation, blockedCoordinate, chunkStart,
                chunkSize, range.getStep(), range.getLogicalStart(),
                range.getLogicalStop(), range.getSourceId(),
                range.getSourceAxis(), range.getDerived());
            inheritRangeAuthority(coordinate, range);
            mapping.map(range.getResult(), coordinate.getResult());
            Value end = nested.create<BroadcastOp>(nestedLocation,
                                                   blockedCoordinate, stop);
            auto coordinatePredicate = FragmentType::get(
                reduce.getContext(), nested.getI1Type(),
                blockedCoordinate.getShape(), blockedCoordinate.getAxisMaps(),
                blockedCoordinate.getValidity(),
                blockedCoordinate.getOwner());
            Value valid = nested.create<CompareOp>(
                nestedLocation, coordinatePredicate, coordinate.getResult(), end,
                ComparePredicate::Lt);
            FailureOr<Value> projected = projectPredicateToFragmentAxis(
                nested, nestedLocation, valid, blockedPredicate,
                plan.reductionAxis);
            if (failed(projected)) {
              bodyFailed = true;
              bodyFailure = "could not project the reduction traversal tail";
              return;
            }
            sourceTail = sourceTail
                             ? Value(nested.create<BinaryOp>(
                                   nestedLocation, blockedPredicate, sourceTail,
                                   *projected, BinaryOperator::LogicalAnd))
                             : *projected;
          }
          auto sourceType = cast<FragmentType>(plan.source.getType());
          auto relation = cast<AxisMapAttr>(sourceType.getAxisMaps()[plan.reductionAxis]);
          Value sourceCoordinate = masterCoordinate;
          if (!plan.ranges.empty()) {
            MakeRangeOp range = plan.ranges.front();
            sourceCoordinate = mapping.lookup(range.getResult());
          }
          if (failed(bindSourceReplay(nested, nestedLocation, plan.source,
                  plan.sourceIdentity, relation.getDimensionId(), plan.reductionAxis,
                  chunkExtent, sourceCoordinate, reduce,
                  PhysicalReplayScope::ValueGraph, {}, mapping, &reuse))) {
            bodyFailed = true;
            bodyFailure = "could not bind the current source snapshot";
            return;
          }
          auto remaining = PhysicalProgramAnalysis(kernel).replayAt(
              plan.source, plan.sourceIdentity, PhysicalReplayScope::ValueGraph,
              true, reduce, mapping, relation.getDimensionId());
          if (!remaining.isReplayable()) {
            bodyFailed = true;
            bodyFailure = "bound source has no complete producer replay proof";
            return;
          }
          for (RootAccess access : accesses[component]) {
            LoadOp load = access.load;
            if (mapping.lookupOrNull(load.getResult()) ||
                !llvm::is_contained(remaining.accesses, load.getOperation())) continue;
            MakeRangeOp range = access.range;
            auto rootType = cast<FragmentType>(load.getResult().getType());
            FragmentType blockedRoot =
                replaceExtent(rootType, access.fragmentAxis, chunkExtent);
            SmallVector<Attribute> coordinateShape(
                range.getResult().getType().getShape().begin(),
                range.getResult().getType().getShape().end());
            coordinateShape[0] = chunkExtent;
            auto blockedCoordinate = FragmentType::get(
                reduce.getContext(), range.getResult().getType().getElementType(),
                ArrayAttr::get(reduce.getContext(), coordinateShape),
                range.getResult().getType().getAxisMaps(),
                range.getResult().getType().getValidity(),
                range.getResult().getType().getOwner());
            Value coordinate = nested.create<MakeRangeOp>(
                nestedLocation, blockedCoordinate, chunkStart, chunkSize,
                range.getStep(), range.getLogicalStart(), range.getLogicalStop(),
                range.getSourceId(), range.getSourceAxis(), range.getDerived());
            inheritRangeAuthority(coordinate.getDefiningOp(), range);
            if (!mapping.lookupOrNull(range.getResult())) {
              auto originalCoordinate = range.getResult().getType();
              auto replayCoordinate = FragmentType::get(
                  reduce.getContext(), originalCoordinate.getElementType(),
                  blockedCoordinate.getShape(), blockedCoordinate.getAxisMaps(),
                  originalCoordinate.getValidity(),
                  originalCoordinate.getOwner());
              Value mappedCoordinate = coordinate;
              if (replayCoordinate != blockedCoordinate) {
                FailureOr<ArrayAttr> reassociation =
                    inferReshapeReassociation(blockedCoordinate,
                                              replayCoordinate);
                if (failed(reassociation)) {
                  bodyFailed = true;
                  bodyFailure =
                      "reduction coordinate has no exact row-major reassociation";
                  return;
                }
                mappedCoordinate = nested.create<ReshapeOp>(
                    nestedLocation, replayCoordinate, coordinate,
                    *reassociation);
              }
              mapping.map(range.getResult(), mappedCoordinate);
            }
            Value end = nested.create<BroadcastOp>(nestedLocation,
                                                   blockedCoordinate, stop);
            auto coordinatePredicate = FragmentType::get(
                reduce.getContext(), nested.getI1Type(),
                blockedCoordinate.getShape(), blockedCoordinate.getAxisMaps(),
                blockedCoordinate.getValidity(),
                blockedCoordinate.getOwner());
            Value coordinateValid = nested.create<CompareOp>(
                nestedLocation, coordinatePredicate, coordinate, end,
                ComparePredicate::Lt);
            auto rootPredicate = FragmentType::get(
                reduce.getContext(), nested.getI1Type(), blockedRoot.getShape(),
                blockedRoot.getAxisMaps(), blockedRoot.getValidity(),
                blockedRoot.getOwner());
            auto projectedValid = predicateForReductionSource(
                nested, nestedLocation, coordinateValid, rootPredicate,
                access.fragmentAxis);
            if (failed(projectedValid)) {
              bodyFailed = true;
              bodyFailure = "source tail cannot adopt its reduction axis";
              return;
            }
            Value valid = *projectedValid;
            auto replayAccessValue = [&](Value value) -> FailureOr<Value> {
              ReplayMaterializationOptions accessOptions = replayOptions;
              if (auto type = dyn_cast<FragmentType>(value.getType())) {
                BroadcastProjection relation =
                    queryAxisProjection(type, rootType);
                if (!relation.isExact())
                  return failure();
                accessOptions.fragmentAxis =
                    relation.targetToSource[access.fragmentAxis];
                if (!accessOptions.fragmentAxis)
                  return mapping.lookupOrDefault(value);
              }
              return materializeReplayedValue(
                  nested, nestedLocation, value, plan.sourceIdentity,
                  chunkExtent, mapping, reduce, accessOptions);
            };
            if (load.getValid()) {
              FailureOr<Value> original = replayAccessValue(load.getValid());
              if (failed(original)) {
                bodyFailed = true;
                bodyFailure = "could not replay source validity";
                return;
              }
              Value originalValid = *original;
              if (originalValid.getType() != rootPredicate)
                originalValid = nested.create<BroadcastOp>(
                    nestedLocation, rootPredicate, originalValid);
              valid = nested.create<BinaryOp>(nestedLocation, rootPredicate,
                                              valid, originalValid,
                                              BinaryOperator::LogicalAnd);
            }
            Value fill;
            if (load.getFill()) {
              FailureOr<Value> replayedFill = replayAccessValue(load.getFill());
              if (failed(replayedFill)) {
                bodyFailed = true;
                bodyFailure = "could not replay source fill";
                return;
              }
              fill = *replayedFill;
              if (fill.getType() != blockedRoot)
                fill = nested.create<BroadcastOp>(nestedLocation, blockedRoot,
                                                  fill);
            } else {
              FailureOr<Value> zero = materializeZeroFragment(
                  nested, nestedLocation, blockedRoot);
              if (failed(zero)) {
                bodyFailed = true;
                bodyFailure = "source element type has no zero fill";
                return;
              }
              fill = *zero;
            }
            SmallVector<Value> coordinates(load.getCoordinates());
            bool cartesian = static_cast<size_t>(llvm::count_if(coordinates, [](Value coordinate) {
              return isa<FragmentType>(coordinate.getType());
            })) == rootType.getShape().size() &&
                llvm::all_of(coordinates, [](Value coordinate) {
                  auto type = dyn_cast<FragmentType>(coordinate.getType());
                  return !type || type.getShape().size() == 1;
                });
            for (unsigned coordinateIndex = 0;
                 coordinateIndex < coordinates.size(); ++coordinateIndex) {
              Value &coordinate = coordinates[coordinateIndex];
              auto originalType = dyn_cast<FragmentType>(coordinate.getType());
              std::optional<unsigned> coordinateAxis;
              if (originalType) {
                if (cartesian) {
                  if (coordinateIndex != access.coordinateIndex)
                    continue;
                  coordinateAxis = 0;
                } else {
                  auto relation = queryAxisProjection(originalType, rootType);
                  if (relation.isExact())
                    coordinateAxis = relation.targetToSource[access.fragmentAxis];
                }
              }
              ReplayMaterializationOptions coordinateOptions = replayOptions;
              coordinateOptions.fragmentAxis = coordinateAxis;
              FailureOr<Value> replayedCoordinate = cartesian
                  ? materializeReplayedValue(nested, nestedLocation, coordinate,
                        plan.sourceIdentity, chunkExtent, mapping, reduce,
                        coordinateOptions)
                  : replayAccessValue(coordinate);
              if (failed(replayedCoordinate)) {
                bodyFailed = true;
                bodyFailure = "could not replay source coordinate";
                return;
              }
              coordinate = *replayedCoordinate;
              auto coordinateType =
                  dyn_cast<FragmentType>(coordinate.getType());
              if (coordinateType && coordinateAxis) {
                FailureOr<Value> projected = projectPhysicalValueToSchema(
                    nested, nestedLocation, coordinate,
                    replaceExtent(coordinateType, *coordinateAxis,
                                  chunkExtent));
                if (failed(projected)) {
                  bodyFailed = true;
                  bodyFailure =
                      "source coordinate cannot adopt the reduction chunk";
                  return;
                }
                coordinate = *projected;
              }
            }
            auto blockedLoad = materializeSourceRead(
                nested, nestedLocation, access.load, access.fragmentAxis, blockedRoot, coordinates,
                valid, fill, coordinate, reduce);
            if (failed(blockedLoad)) {
              bodyFailed = true;
              bodyFailure = "could not preserve the original reduction read";
              return;
            }
            mapping.map(load.getResult(), *blockedLoad);
            if (!sourceTail) {
              auto tail = predicateForReductionSource(
                  nested, nestedLocation, coordinateValid, blockedPredicate,
                  plan.reductionAxis);
              if (failed(tail)) {
                bodyFailed = true;
                bodyFailure = "source tail cannot adopt the producer reduction axis";
                return;
              }
              sourceTail = *tail;
            }
          }
          replayOptions.fragmentAxis = plan.reductionAxis;
          FailureOr<Value> replayed = materializeReplayedValue(
              nested, nestedLocation, plan.source, plan.sourceIdentity,
              chunkExtent, mapping, reduce, replayOptions);
          if (failed(replayed) || !sourceTail) {
            if (failed(replayed)) {
              bodyFailed = true;
              bodyFailure = "could not replay load-rooted pure producer graph";
              return;
            }
            sourceTail = nested.create<BroadcastOp>(
                nestedLocation, blockedPredicate, sharedTail);
          }
          Value identity = identities[component];
          if (auto fragment = dyn_cast<FragmentType>(identity.getType());
              fragment && fragment != blockedSource) {
            FragmentType expanded = blockedSource;
            auto unit = expression(reduce.getContext(), PhysicalExprKind::Constant, 1);
            for (int64_t axis : reduce.getAxes())
              expanded = replaceExtent(expanded, axis, unit);
            FailureOr<ArrayAttr> relation =
                inferReshapeReassociation(fragment, expanded);
            if (failed(relation)) {
              bodyFailed = true;
              bodyFailure = "reduction identity has no free-axis projection";
              return;
            }
            identity = nested.create<ReshapeOp>(
                nestedLocation, expanded, identity, *relation);
          }
          if (identity.getType() != blockedSource)
            identity = nested.create<BroadcastOp>(nestedLocation, blockedSource,
                                                  identity);
          blockedSources.push_back(nested.create<SelectOp>(
              nestedLocation, blockedSource, sourceTail, *replayed, identity));
        }
        if (bodyFailed)
          return;

        if (vectorAccumulation) {
          SmallVector<Value> combineArguments(carries.begin(), carries.end());
          llvm::append_range(combineArguments, blockedSources);
          llvm::append_range(combineArguments, captures);
          auto combined = inlinePureRegion(
              nested, vectorCombine, combineArguments, bodyFailure);
          if (failed(combined)) {
            bodyFailed = true;
            return;
          }
          nested.create<scf::YieldOp>(nestedLocation, *combined);
          return;
        }

        auto chunkReduce = nested.create<ReduceOp>(nestedLocation, blockedSources,
            identities, captures, reduce.getAxes());
        if (Attribute origin = reduce->getAttr(originAttr))
          chunkReduce->setAttr(originAttr, origin);
        IRMapping regionMapping;
        reduce.getCombine().cloneInto(&chunkReduce.getCombine(), regionMapping);

        SmallVector<Value> combineArguments(carries.begin(), carries.end());
        combineArguments.append(chunkReduce.getResults().begin(),
                                chunkReduce.getResults().end());
        combineArguments.append(captures.begin(), captures.end());
        FailureOr<SmallVector<Value>> combined = inlinePureRegion(
            nested, chunkReduce.getCombine(), combineArguments, bodyFailure);
        if (failed(combined)) {
          bodyFailed = true;
          return;
        }
        nested.create<scf::YieldOp>(nestedLocation, *combined);
      };
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(loop.getBody());
  buildBody(bodyBuilder, location, loop.getInductionVar(), loop.getRegionIterArgs());
  if (Attribute origin = reduce->getAttr(originAttr))
    loop->setAttr(originAttr, origin);
  if (bodyFailed) {
    loop.erase();
    return reduce.emitOpError(
        "runtime reduction could not be materialized in the chunk loop: ")
           << bodyFailure;
  }
  loop->setAttr(reductionSourcesAttr, reductionSources(reduce));

  SmallVector<Value> realizedResults(loop.getResults().begin(),
                                     loop.getResults().end());
  if (vectorAccumulation) {
    auto finalReduce = builder.create<ReduceOp>(location, realizedResults,
        identities, captures, reduce.getAxes());
    if (Attribute origin = reduce->getAttr(originAttr))
      finalReduce->setAttr(originAttr, origin);
    IRMapping regionMapping;
    reduce.getCombine().cloneInto(&finalReduce.getCombine(), regionMapping);
    realizedResults.assign(finalReduce.getResults().begin(),
                           finalReduce.getResults().end());
  }
  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), realizedResults))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  eraseDeadPhysicalValues(kernel);
  return success();
}

namespace {

bool isolatedReductionUpdate(scf::ForOp loop, ReduceOp reference,
                             SmallVectorImpl<Value> &right,
                             llvm::SmallPtrSetImpl<Operation *> &matched) {
  if (!loop || !loop->hasAttr(reductionSourcesAttr) ||
      loop.getNumRegionIterArgs() != reference.getSources().size())
    return false;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!matchReductionCombine(reference, loop.getBody(),
                             loop.getRegionIterArgs(), right,
                             yield.getOperands(), matched))
    return false;
  for (Value carry : loop.getRegionIterArgs())
    for (Operation *user : carry.getUsers())
      if (!matched.contains(user))
        return false;
  for (Operation *operation : matched)
    for (Operation *user : operation->getUsers())
      if (user != yield && !matched.contains(user))
        return false;
  return true;
}

PhysicalExprAttr savedCollectivePayloadWork(scf::ForOp loop,
    TypeRange accumulatorTypes, func::FuncOp kernel) {
  auto lower = queryLaunchExpression(loop.getLowerBound());
  auto upper = queryLaunchExpression(loop.getUpperBound());
  auto step = queryLaunchExpression(loop.getStep());
  auto payload = nominalPayloadWords(accumulatorTypes);
  if (!lower || !upper || !step || !payload) return {};
  auto context = loop.getContext();
  auto distance = expression(context, PhysicalExprKind::Subtract, 0, {}, {upper, lower});
  auto distanceBounds = queryPhysicalExpressionRange(distance, kernel);
  auto stepBounds = queryPhysicalExpressionRange(step, kernel);
  if (!distanceBounds || distanceBounds->smin().isNegative() ||
      !stepBounds || !stepBounds->smin().isStrictlyPositive()) return {};
  auto trips = expression(context, PhysicalExprKind::CeilDiv, 0, {}, {distance, step});
  auto one = expression(context, PhysicalExprKind::Constant, 1);
  auto zero = expression(context, PhysicalExprKind::Constant, 0);
  auto repeated = expression(context, PhysicalExprKind::Subtract, 0, {}, {trips, one});
  // Empty and one-trip loops save no repeated collective work.
  repeated = expression(context, PhysicalExprKind::Maximum, 0, {}, {repeated, zero});
  auto saved = expression(context, PhysicalExprKind::Multiply, 0, {}, {repeated, payload});
  auto bounds = queryPhysicalExpressionRange(saved, kernel);
  return bounds && !bounds->smin().isNegative() ? saved : PhysicalExprAttr();
}

} // namespace

bool hoistNestedReduction(scf::ForOp outer, func::FuncOp kernel) {
  if (!outer->hasAttr(reductionSourcesAttr) ||
      outer.getNumRegionIterArgs() == 0)
    return false;
  unsigned count = outer.getNumRegionIterArgs();
  ReduceOp reference;
  llvm::SmallPtrSet<Operation *, 32> updates;
  for (ReduceOp candidate : outer.getBody()->getOps<ReduceOp>()) {
    if (candidate.getNumResults() != count)
      continue;
    SmallVector<Value> right(candidate.getResults());
    llvm::SmallPtrSet<Operation *, 32> candidateUpdates;
    if (isolatedReductionUpdate(outer, candidate, right, candidateUpdates)) {
      reference = candidate;
      updates = std::move(candidateUpdates);
      break;
    }
  }
  if (!reference)
    return false;
  SmallVector<Value> identities;
  DominanceInfo dominance(kernel);
  for (Value init : outer.getInitArgs()) {
    FailureOr<Value> identity = scalarSource(init);
    if (failed(identity) || !dominance.dominates(*identity, outer))
      return false;
    identities.push_back(*identity);
  }

  SmallVector<ReduceOp> reductions;
  SmallVector<Value> sources(reference.getResults());
  while (auto reduce = sources.front().getDefiningOp<ReduceOp>()) {
    if (reduce->getBlock() != outer.getBody() ||
        reduce.getSources().size() != count || reduce.getIdentities().size() != count ||
        reduce.getCaptures().size() != 0 || reduce.getNumResults() != count ||
        !llvm::equal(sources, reduce.getResults()))
      return false;
    Block &combine = reduce.getCombine().front();
    SmallVector<Value> right(combine.getArguments().drop_front(count));
    llvm::SmallPtrSet<Operation *, 32> matched;
    if (!matchReductionCombine(reference, &combine,
                               combine.getArguments().take_front(count), right,
                               cast<YieldOp>(combine.getTerminator()).getValues(),
                               matched))
      return false;
    for (unsigned i = 0; i < count; ++i) {
      if (!sameScalarValue(reduce.getIdentities()[i], identities[i]))
        return false;
      for (Operation *user : reduce.getResult(i).getUsers())
        if (!updates.contains(user) &&
            (reductions.empty() || user != reductions.back()))
          return false;
    }
    reductions.push_back(reduce);
    sources.assign(reduce.getSources().begin(), reduce.getSources().end());
  }
  SmallVector<Type> accumulatorTypes;
  for (unsigned i = 0; i < count; ++i) {
    auto type = dyn_cast<FragmentType>(sources[i].getType());
    if (!type || !llvm::all_of(type.getShape(), [](Attribute extent) {
          return isShapeBound(cast<PhysicalExprAttr>(extent));
        }))
      return false;
    accumulatorTypes.push_back(type);
  }
  Region scalarCombine, vectorCombine;
  std::string reason;
  if (failed(scalarizeElementwiseCallback(reference.getCombine(), scalarCombine)) ||
      failed(liftCombineRegion(scalarCombine, vectorCombine,
                                      accumulatorTypes, reason)))
    return false;

  SmallVector<Operation *> ignored(updates.begin(), updates.end());
  for (ReduceOp reduce : reductions) ignored.push_back(reduce);
  SmallVector<Type> additionalPayloads;
  for (Operation &operation : vectorCombine.front().without_terminator()) {
    if (isa<BroadcastOp, SplatOp, ReshapeOp, MakeRecordOp, ExtractOp>(&operation))
      continue;
    for (Type type : operation.getResultTypes())
      if (isa<FragmentType, RecordType>(type)) additionalPayloads.push_back(type);
  }
  auto workingSet = nominalLoopWorkingSet(outer, accumulatorTypes, ignored,
                                          additionalPayloads, sources);
  auto bounds = workingSet ? queryPhysicalExpressionRange(workingSet, kernel)
                           : std::nullopt;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!bounds || bounds->smin().isNegative() || !capabilities ||
      capabilities.getRegistersPerUnit() <= 0)
    return false;
  int64_t budget = capabilities.getRegistersPerUnit();
  auto savedWork = savedCollectivePayloadWork(outer, accumulatorTypes, kernel);
  auto savedBounds = savedWork ? queryPhysicalExpressionRange(savedWork, kernel)
                              : std::nullopt;
  bool alwaysWorthHoisting = savedBounds &&
      savedBounds->smin().getSExtValue() >= bounds->smax().getSExtValue();
  bool canSaveEnough = savedBounds &&
      savedBounds->smax().getSExtValue() >= bounds->smin().getSExtValue();
  if (bounds->smin().getSExtValue() > budget && !canSaveEnough) return false;
  bool guarded = bounds->smax().getSExtValue() > budget && !alwaysWorthHoisting;
  if (guarded) {
    // Both versions keep their original loop allocation scope. Nevertheless,
    // cloning allocations would duplicate the physical resource identities.
    auto effects = getEffectsRecursively(outer);
    if (!effects || llvm::any_of(*effects, [](const auto &effect) {
          return isa<MemoryEffects::Allocate>(effect.getEffect());
        })) return false;
    for (Operation *current = outer; Operation *parent = current->getParentOp();
         current = parent) {
      auto branch = dyn_cast<scf::IfOp>(parent);
      if (!branch || current->getParentRegion() != &branch.getElseRegion()) continue;
      bool includesBudget = false, includesSavings = false;
      SmallVector<Value> pending{branch.getCondition()};
      while (!pending.empty()) {
        Value value = pending.pop_back_val();
        if (auto disjunction = value.getDefiningOp<BinaryOp>();
            disjunction && disjunction.getOperatorKind() == BinaryOperator::LogicalOr) {
          pending.push_back(disjunction.getLhs());
          pending.push_back(disjunction.getRhs());
          continue;
        }
        auto condition = value.getDefiningOp<CompareOp>();
        if (!condition) continue;
        auto lhs = queryLaunchExpression(condition.getLhs());
        auto rhs = queryLaunchExpression(condition.getRhs());
        includesBudget |= condition.getPredicate() == ComparePredicate::Le &&
            lhs == workingSet && IndexRelations().constant(condition.getRhs()) == budget;
        includesSavings |= savedWork &&
            ((condition.getPredicate() == ComparePredicate::Ge &&
              lhs == savedWork && rhs == workingSet) ||
             (condition.getPredicate() == ComparePredicate::Le &&
              lhs == workingSet && rhs == savedWork));
      }
      if (includesBudget && (!canSaveEnough || includesSavings)) return false;
    }
  }

  // These are compiler-created ordinary-reduction traversals. Keep their
  // existing tile, loads, masks and effects, but carry the tile through the
  // loop when its nominal working set fits or the eliminated repeated
  // collective payload work covers that footprint. This is a scheduling cost
  // choice; the provider owns native allocation and the collective's tree.
  // The use checks exclude observable partial summaries and ordered carries.
  OpBuilder builder(outer);
  scf::IfOp choice;
  if (guarded) {
    Value words = builder.create<PhysicalExprOp>(outer.getLoc(),
        builder.getIndexType(), workingSet);
    Value limit = builder.create<arith::ConstantIndexOp>(outer.getLoc(), budget);
    Value fits = builder.create<CompareOp>(outer.getLoc(), builder.getI1Type(),
                                          words, limit, ComparePredicate::Le);
    if (canSaveEnough) {
      Value saved = builder.create<PhysicalExprOp>(outer.getLoc(),
          builder.getIndexType(), savedWork);
      Value worthHoisting = builder.create<CompareOp>(outer.getLoc(), builder.getI1Type(),
          saved, words, ComparePredicate::Ge);
      fits = builder.create<BinaryOp>(outer.getLoc(), builder.getI1Type(),
                                      fits, worthHoisting, BinaryOperator::LogicalOr);
    }
    choice = builder.create<scf::IfOp>(outer.getLoc(), outer.getResultTypes(), fits, true);
    builder.setInsertionPointToStart(choice.thenBlock());
  }
  SmallVector<Value> initial;
  for (auto [source, identity] : llvm::zip(sources, identities))
    initial.push_back(builder.create<SplatOp>(
        outer.getLoc(), cast<FragmentType>(source.getType()), identity));
  auto replacement = builder.create<scf::ForOp>(
      outer.getLoc(), outer.getLowerBound(), outer.getUpperBound(),
      outer.getStep(), initial);
  replacement->setAttrs(outer->getAttrs());
  IRMapping mapping;
  mapping.map(outer.getInductionVar(), replacement.getInductionVar());
  builder.setInsertionPointToStart(replacement.getBody());
  for (Operation &operation : outer.getBody()->without_terminator()) {
    if (updates.contains(&operation) ||
        llvm::any_of(reductions, [&](ReduceOp reduce) {
          return reduce.getOperation() == &operation;
        }))
      continue;
    builder.clone(operation, mapping);
  }
  SmallVector<Value> arguments(replacement.getRegionIterArgs());
  for (Value source : sources)
    arguments.push_back(mapping.lookupOrDefault(source));
  FailureOr<SmallVector<Value>> carried =
      inlinePureRegion(builder, vectorCombine, arguments, reason);
  assert(succeeded(carried) && "validated reduction combine failed to inline");
  builder.create<scf::YieldOp>(outer.getLoc(), *carried);

  builder.setInsertionPointAfter(replacement);
  SmallVector<Value> results(replacement.getResults());
  for (ReduceOp reduce : llvm::reverse(reductions)) {
    IRMapping reduceMapping;
    for (unsigned i = 0; i < count; ++i) {
      reduceMapping.map(reduce.getSources()[i], results[i]);
      Value projectedIdentity = identities[i];
      Value originalIdentity = reduce.getIdentities()[i];
      if (auto fragment = dyn_cast<FragmentType>(originalIdentity.getType()))
        projectedIdentity = builder.create<SplatOp>(
            reduce.getLoc(), fragment, projectedIdentity);
      reduceMapping.map(originalIdentity, projectedIdentity);
    }
    Operation *cloned = builder.clone(*reduce, reduceMapping);
    results.assign(cloned->getResults().begin(), cloned->getResults().end());
  }
  if (choice) {
    builder.create<scf::YieldOp>(outer.getLoc(), results);
    outer.replaceAllUsesWith(choice.getResults());
    outer->moveBefore(choice.elseBlock(), choice.elseBlock()->end());
    builder.setInsertionPointAfter(outer);
    builder.create<scf::YieldOp>(outer.getLoc(), outer.getResults());
  } else {
    outer.replaceAllUsesWith(results);
    outer.erase();
  }
  return true;
}

bool sinkReductionIntoSourceIf(ReduceOp reduce, func::FuncOp kernel) {
  auto sources = reduce.getSources();
  auto branch = sources.empty() ? scf::IfOp()
                               : sources.front().getDefiningOp<scf::IfOp>();
  if (!branch || branch.getElseRegion().empty() ||
      branch->getBlock() != reduce->getBlock() ||
      sources.size() != reduce.getNumResults())
    return false;
  SmallVector<unsigned> resultIndices;
  for (Value source : sources) {
    auto result = dyn_cast<OpResult>(source);
    if (!result || result.getOwner() != branch.getOperation() ||
        !result.hasOneUse() ||
        llvm::is_contained(resultIndices, result.getResultNumber()))
      return false;
    resultIndices.push_back(result.getResultNumber());
  }
  DominanceInfo dominance(kernel);
  SmallVector<Operation *> identities;
  for (Value value : llvm::concat<const Value>(reduce.getIdentities(), reduce.getCaptures())) {
    if (dominance.dominates(value, branch.getOperation()))
      continue;
    Operation *producer = value.getDefiningOp();
    if (!producer || !isa<arith::ConstantOp, SplatOp>(producer) ||
        !llvm::all_of(producer->getOperands(), [&](Value operand) {
          return dominance.dominates(operand, branch.getOperation());
        }))
      return false;
    if (!llvm::is_contained(identities, producer))
      identities.push_back(producer);
  }
  OpBuilder builder(branch);
  IRMapping captures;
  for (Operation *identity : identities)
    builder.clone(*identity, captures);
  // Keep conditionally executed reads in their original branch. The pure
  // reduction consumes exactly that branch's yielded values before its exit.
  for (Region *region : {&branch.getThenRegion(), &branch.getElseRegion()}) {
    auto yield = cast<scf::YieldOp>(region->front().getTerminator());
    SmallVector<Value> yielded(yield.getOperands());
    IRMapping mapping(captures);
    for (auto [source, index] : llvm::zip(sources, resultIndices))
      mapping.map(source, yielded[index]);
    builder.setInsertionPoint(yield);
    auto reduced = cast<ReduceOp>(builder.clone(*reduce.getOperation(), mapping));
    for (auto [index, result] : llvm::zip(resultIndices, reduced.getResults()))
      yielded[index] = result;
    yield->setOperands(yielded);
  }
  for (auto [index, reduced] : llvm::zip(resultIndices, reduce.getResults())) {
    Value result = branch.getResult(index);
    result.setType(reduced.getType());
    reduced.replaceAllUsesWith(result);
  }
  reduce.erase();
  return true;
}

} // namespace intent::gpu::reduction
