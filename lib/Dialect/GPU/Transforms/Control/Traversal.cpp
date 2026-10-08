#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;

namespace intent::gpu {

scf::ForOp createTraversalLoop(
    OpBuilder &builder, Location location, Value lower, Value upper, Value step,
    ValueRange initialValues,
    llvm::function_ref<void(OpBuilder &, Location, Value, ValueRange)> buildBody) {
  auto loop = builder.create<scf::ForOp>(
      location, lower, upper, step, initialValues,
      [](OpBuilder &, Location, Value, ValueRange) {});
  OpBuilder bodyBuilder(loop.getBody(), loop.getBody()->begin());
  buildBody(bodyBuilder, location, loop.getInductionVar(), loop.getRegionIterArgs());
  return loop;
}

static FailureOr<PhysicalExprAttr>
boundedCapacity(func::FuncOp kernel, ParameterAttr chunk,
                PhysicalExprAttr capacity, bool preserveCandidateDomain) {
  auto expression = [&](PhysicalExprKind kind, int64_t value = 0,
                        StringRef symbol = {}, ArrayRef<Attribute> operands = {}) {
    return PhysicalExprAttr::get(chunk.getContext(), kind,
                                 value, kind == PhysicalExprKind::Parameter
                                     ? Attribute(ParameterRefAttr::get(chunk.getContext(), StringAttr::get(chunk.getContext(), symbol)))
                                     : Attribute(StringAttr::get(chunk.getContext(), symbol)),
                                 ArrayAttr::get(chunk.getContext(), operands));
  };
  auto extent = expression(PhysicalExprKind::Parameter, 0,
                           chunk.getName().getValue());
  if (!capacity || !isShapeBound(capacity) || chunk.isDeferred())
    return extent;
  int64_t maximum =
      *llvm::max_element(chunk.getCandidates().asArrayRef());
  if (maximum > (int64_t{1} << 62))
    return extent;
  if (auto bound = constantPhysicalExpression(capacity);
      bound && *bound >= maximum)
    return extent;
  auto positive = expression(PhysicalExprKind::Maximum, 0, {},
                             {capacity, expression(PhysicalExprKind::Constant, 1)});
  auto bounded = expression(PhysicalExprKind::Minimum, 0, {},
                            {positive, expression(PhysicalExprKind::Constant, maximum)});
  auto padded = expression(PhysicalExprKind::NextPowerOfTwo, 0, {}, {bounded});
  SmallVector<int64_t> candidates;
  if (preserveCandidateDomain) {
    candidates.assign(chunk.getCandidates().asArrayRef().begin(),
                      chunk.getCandidates().asArrayRef().end());
    llvm::sort(candidates);
    if (auto bound = constantPhysicalExpression(bounded)) {
      auto selected = llvm::lower_bound(candidates, *bound);
      if (selected == candidates.end()) return failure();
      padded = expression(PhysicalExprKind::Constant, *selected);
    }
  } else {
    for (int64_t candidate = 1; candidate < maximum; candidate *= 2)
      candidates.push_back(candidate);
    candidates.push_back(llvm::PowerOf2Ceil(static_cast<uint64_t>(maximum)));
  }
  if (!isCompileTimePhysicalExpr(padded) ||
      (preserveCandidateDomain && !constantPhysicalExpression(bounded))) {
    // Coverage is bound before candidate selection and cannot depend on another
    // parameter. A deferred parameter's coverageBound is only its required size,
    // not the value selected from its domain, so substituting that bound would
    // change this capacity. Leave optional clipping unused in this case.
    AttrTypeWalker dependencies;
    dependencies.addWalk([](ParameterRefAttr) { return WalkResult::interrupt(); });
    if (dependencies.walk(bounded).wasInterrupted()) return extent;
    ParameterAttr capacityParameter;
    // Host-dependent capacity is bound once before candidate selection. Keep
    // runtime dimensions out of fragment types, without duplicating the host
    // evaluator or baking a particular invocation into the shared program.
    for (Attribute attribute : getParameterDeclarations(kernel)) {
      auto declaration = cast<ParameterAttr>(attribute);
      if (declaration.isDeferred() &&
          declaration.getCategory() == ParameterCategory::Coverage &&
          declaration.getBinding().getCoverageBound() == bounded &&
          llvm::equal(declaration.getCandidates().asArrayRef(), candidates)) {
        capacityParameter = declaration;
        break;
      }
    }
    if (!capacityParameter) {
      std::string stem = (chunk.getName().getValue() + "_CAPACITY").str();
      std::string name = stem;
      for (unsigned suffix = 1; lookupParameter(kernel, StringAttr::get(kernel.getContext(), name)); ++suffix)
        name = stem + "_" + std::to_string(suffix);
      auto binding = ParameterBindingAttr::get(kernel.getContext(), {}, {},
                                               bounded, {}, false, false);
      auto reference = getOrCreatePhysicalParameter(kernel, name,
          ParameterRole::FullCoverage, ParameterCategory::Coverage,
          /*elementBitWidth=*/0, candidates, binding);
      if (failed(reference)) return failure();
      capacityParameter = lookupParameter(kernel, *reference);
    }
    padded = expression(PhysicalExprKind::Parameter, 0,
                        capacityParameter.getName().getValue());
  }
  return expression(PhysicalExprKind::Minimum, 0, {}, {extent, padded});
}

FailureOr<PhysicalExprAttr> boundedTraversalChunk(ParameterAttr chunk,
                                                 MakeRangeOp range) {
  return boundedCapacity(range->getParentOfType<func::FuncOp>(), chunk,
                         queryLogicalRangeCapacity(range), false);
}

FailureOr<PhysicalExprAttr>
boundedOwnershipExtent(func::FuncOp kernel, ParameterAttr parameter,
                       PhysicalExprAttr logicalCapacity) {
  return boundedCapacity(kernel, parameter, logicalCapacity, true);
}

LogicalResult realizeFullCoverageDimension(func::FuncOp kernel, Value source,
                                           unsigned fragmentAxis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || fragmentAxis >= fragment.getShape().size())
    return failure();
  auto mapping =
      dyn_cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
  if (!mapping || mapping.getDimensionId() <= 0)
    return failure();
  const int64_t dimension = mapping.getDimensionId();
  auto currentExtent =
      cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact ranges = analysis.axisRanges(source, fragmentAxis);
  if (ranges.roots.empty())
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps()))
      if (cast<AxisMapAttr>(attribute).getDimensionId() == dimension &&
          fragment.getShape()[axis] != currentExtent)
        return emitError(source.getLoc(),
                         "full coverage has no range authority for distinct physical occurrences");
  int64_t coverageDimension = dimension;
  bool subregion = llvm::any_of(ranges.roots, [](MakeRangeOp range) {
    return range->hasAttr(sourceSubregionAttr);
  });
  PhysicalExprAttr rangeCapacity;
  if (subregion) {
    FailureOr<MakeRangeOp> authority = queryExactLogicalRange(ranges);
    if (failed(authority))
      return kernel.emitError(
          "subregion full coverage has no exact logical range authority");
    rangeCapacity = queryLogicalRangeCapacity(*authority);
    std::optional<int64_t> parent;
    for (MakeRangeOp range : ranges.roots) {
      auto bound = range->getAttrOfType<IntegerAttr>(sourceSubregionAttr);
      if (!bound || bound.getInt() <= 0 ||
          (parent && *parent != bound.getInt()) ||
          (!rangeCapacity &&
           !queryNonNegativeIndexUpperBound(range.getLogicalStart())) ||
          !isUnitStepRange(range))
        return range.emitOpError(
            "subregion full coverage has no proven nonnegative parent-bounded traversal");
      parent = bound.getInt();
    }
    if (!rangeCapacity)
      coverageDimension = *parent;
  }

  Value runtimeDimension;
  bool coverageIsRangeCapacity = static_cast<bool>(rangeCapacity);
  if (rangeCapacity) {
    OpBuilder builder(&kernel.front(), kernel.front().begin());
    runtimeDimension = builder.create<PhysicalExprOp>(
        source.getLoc(), builder.getIndexType(), rangeCapacity);
  }
  if (!coverageIsRangeCapacity)
    runtimeDimension = resolveDimension(kernel, coverageDimension);

  std::optional<int64_t> staticDimension;
  for (BlockArgument argument : kernel.getArguments()) {
    if (coverageIsRangeCapacity)
      break;
    auto view = dyn_cast<ViewType>(argument.getType());
    if (!view)
      continue;
    for (auto [axis, identity] :
         llvm::enumerate(view.getLayout().getDimensionIds().asArrayRef())) {
      if (identity != coverageDimension)
        continue;
      auto extent = cast<PhysicalExprAttr>(
          view.getLayout().getExtents()[axis]);
      if (extent.getKind() !=
          PhysicalExprKind::Constant)
        continue;
      if (staticDimension && *staticDimension != extent.getValue())
        return kernel.emitError(
            "logical dimension has conflicting static ABI extents");
      staticDimension = extent.getValue();
    }
  }
  if (!runtimeDimension && staticDimension &&
      currentExtent.getKind() ==
          PhysicalExprKind::Constant &&
      currentExtent.getValue() == *staticDimension &&
      llvm::isPowerOf2_64(*staticDimension))
    return success();
  if (!runtimeDimension) {
    FailureOr<MakeRangeOp> authority = queryExactLogicalRange(ranges);
    PhysicalExprAttr capacity = succeeded(authority)
                                   ? queryLogicalRangeCapacity(*authority)
                                   : PhysicalExprAttr();
    if (capacity) {
      OpBuilder builder(&kernel.front(), kernel.front().begin());
      runtimeDimension = builder.create<PhysicalExprOp>(
          source.getLoc(), builder.getIndexType(), capacity);
      coverageIsRangeCapacity = true;
    }
  }
  if (!runtimeDimension)
    return emitError(source.getLoc(),
                     "full-coverage physicalization has no runtime or static dimension authority")
           << "; dimension=" << dimension << "; source=" << source;
  PhysicalExprAttr coverageBound = queryLaunchExpression(runtimeDimension);
  if (!coverageBound)
    return kernel.emitError(
        "full-coverage physicalization has no launch-visible extent expression");
  // A range capacity already bounds its member count. Unlike a parent extent,
  // it is not an absolute coordinate against which to compare logicalStop.
  if (subregion && !coverageIsRangeCapacity)
    for (MakeRangeOp range : ranges.roots) {
      auto end = queryNonNegativeIndexUpperBound(range.getLogicalStop());
      auto endConstant = end ? constantPhysicalExpression(end) : std::nullopt;
      auto coverageConstant = constantPhysicalExpression(coverageBound);
      bool bounded = end &&
          (end == coverageBound ||
           (endConstant && coverageConstant && *endConstant <= *coverageConstant));
      if (!samePhysicalScalarExpression(range.getLogicalStop(), runtimeDimension) &&
          !bounded)
        return range.emitOpError(
            "subregion full coverage has no proven stop bound");
    }

  if (!staticDimension)
    if (auto value = dyn_cast_or_null<IntegerAttr>(
            UniformValueAnalysis(describeUniformValue).evaluate(runtimeDimension)))
      staticDimension = value.getInt();
  if (!staticDimension && coverageBound.getKind() ==
                              PhysicalExprKind::Constant)
    staticDimension = coverageBound.getValue();
  if (staticDimension && *staticDimension >= 0) {
    uint64_t size = llvm::PowerOf2Ceil(
        static_cast<uint64_t>(std::max<int64_t>(*staticDimension, 1)));
    if (size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return kernel.emitError("full-coverage extent exceeds the index range");
    auto covered = PhysicalExprAttr::get(kernel.getContext(),
        PhysicalExprKind::Constant, size,
        StringAttr::get(kernel.getContext(), ""), ArrayAttr::get(kernel.getContext(), {}));
    if (currentExtent == covered &&
        analysis.axisRealization(source, fragmentAxis).physicalized)
      return success();
    OpBuilder builder(&kernel.front(), kernel.front().begin());
    Value physicalExtent = builder.create<arith::ConstantIndexOp>(source.getLoc(), size);
    llvm::SmallDenseSet<int64_t> rangeDimensions;
    for (MakeRangeOp range : ranges.roots) {
      FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
      if (failed(rangeDimension))
        return range.emitOpError("full coverage has no range dimension authority");
      rangeDimensions.insert(*rangeDimension);
      if (failed(retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                                      covered, *rangeDimension)))
        return failure();
    }
    if (ranges.roots.empty() &&
        failed(retargetDimensionExtent(source, dimension, covered)))
      return failure();
    if (rangeDimensions.empty())
      rangeDimensions.insert(dimension);
    for (int64_t rangeDimension : rangeDimensions)
      if (failed(bindFullCoverageDimension(kernel, rangeDimension, physicalExtent)))
        return failure();
    return success();
  }

  ParameterAttr parameter;
  if (currentExtent.getKind() ==
      PhysicalExprKind::Parameter) {
    FailureOr<ParameterAttr> declaration =
        queryParameterBySymbol(kernel, currentExtent.getParameterReference().getName());
    if (succeeded(declaration)) {
      auto covered = declaration->getBinding().getDimension();
      if (declaration->isDeferred() && covered && covered.getInt() == coverageDimension &&
          declaration->getRole() ==
              ParameterRole::FullCoverage)
        parameter = *declaration;
    }
  }
  if (!parameter) {
    for (Attribute attribute : getParameterDeclarations(kernel)) {
      auto candidate = cast<ParameterAttr>(attribute);
      auto covered = candidate.getBinding().getDimension();
      if (!candidate.isDeferred() || !covered || covered.getInt() != coverageDimension ||
          candidate.getRole() !=
              ParameterRole::FullCoverage)
        continue;
      if (parameter && parameter != candidate) {
        parameter = ParameterAttr();
        continue;
      }
      parameter = candidate;
    }
  }
  if (parameter &&
      currentExtent.getKind() ==
          PhysicalExprKind::Parameter &&
      currentExtent.getParameterReference().getName() == parameter.getName() &&
      parameter.getRole() ==
          ParameterRole::FullCoverage) {
    parameter = parameter.withBinding(parameter.getBinding().withCoverageBound(coverageBound));
    if (failed(updateParameter(kernel, parameter))) return failure();
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact() || !binding.dimension ||
        *binding.dimension != coverageDimension)
      return kernel.emitOpError(
          "full-coverage parameter lost its typed dimension authority");
    OpBuilder entry(&kernel.front(), kernel.front().begin());
    if (failed(bindFullCoverageDimension(kernel, dimension,
          materializeParameter(entry, source.getLoc(), parameter.getReference()))))
      return kernel.emitError(
          "existing full-coverage decision could not preserve access validity");
    return success();
  }
  static constexpr int64_t candidates[] = {
      1,    2,    4,     8,     16,    32,    64,    128,   256,
      512,  1024, 2048,  4096,  8192,  16384, 32768, 65536};
  if (!parameter) {
    std::string name = ("FULL_D" + Twine(coverageDimension)).str();
    if (auto existing = lookupParameter(kernel, StringAttr::get(kernel.getContext(), name))) {
      InFlightDiagnostic diagnostic = kernel.emitError(
          "full-coverage parameter name is already owned by another decision");
      diagnostic << "; role=" << stringifyParameterRole(existing.getRole());
      if (auto bound = existing.getBinding().getDimension())
        diagnostic << ", dimension=" << bound.getInt();
      return failure();
    }
    OpBuilder builder(&kernel.getBody().front(),
                      kernel.getBody().front().begin());
    auto schema = ParameterAttr::get(
        kernel.getContext(), builder.getStringAttr(name), builder.getIndexType(),
        ParameterRole::FullCoverage,
        ParameterCategory::Coverage,
        /*elementBitWidth=*/0,
        DenseI64ArrayAttr::get(kernel.getContext(), candidates),
        ConfigurationBindingPhase::Deferred,
        ParameterBindingAttr::get(kernel.getContext(), builder.getI64IntegerAttr(coverageDimension),
                                 {}, coverageBound, {}, false, false));
    if (failed(declareParameter(kernel, schema))) return failure();
    parameter = schema;
  } else {
    parameter = parameter.withCandidates(DenseI64ArrayAttr::get(kernel.getContext(), candidates));
  }
  parameter = parameter.withPhase(ConfigurationBindingPhase::Deferred).withBinding(
      parameter.getBinding().withDimension(IntegerAttr::get(
          IntegerType::get(kernel.getContext(), 64), coverageDimension))
          .withCoverageBound(coverageBound));
  if (failed(updateParameter(kernel, parameter))) return failure();

  auto covered = PhysicalExprAttr::get(
      kernel.getContext(),
      PhysicalExprKind::Parameter, 0,
      parameter.getReference(),
      ArrayAttr::get(kernel.getContext(), {}));
  if (!ranges.roots.empty() && failed(queryExactLogicalRange(ranges)))
    return kernel.emitError(
        "full-coverage source has ambiguous physical range authority");
  if (ranges.state == PhysicalFactState::Unknown || ranges.roots.empty()) {
    PhysicalReplayFact replay = analysis.replayability(
        source, sourceAxisIdentity(mapping), PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/false);
    if (!ranges.roots.empty() || !replay.isReplayable()) {
      InFlightDiagnostic diagnostic = kernel.emitError(
          "full-coverage source has no exact physical range authority");
      for (Operation *blocker : ranges.blockers)
        diagnostic << "; blocker=" << blocker->getName();
      for (Operation *blocker : replay.blockers)
        diagnostic << "; replay_blocker=" << blocker->getName();
      return failure();
    }
  }
  for (MakeRangeOp range : ranges.roots) {
    FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
    if (failed(rangeDimension) || *rangeDimension != dimension)
      return range.emitOpError(
          "full-coverage range does not cover the selected logical dimension");
    if (failed(retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                                    covered, *rangeDimension)))
      return failure();
  }
  if (ranges.roots.empty() &&
      failed(retargetDimensionExtent(source, dimension, covered)))
    return failure();
  OpBuilder entry(&kernel.front(), kernel.front().begin());
  if (failed(bindFullCoverageDimension(kernel, dimension,
        materializeParameter(entry, source.getLoc(), parameter.getReference()))))
    return kernel.emitError(
        "full-coverage decision could not preserve access validity");
  return success();
}

LogicalResult bindFullCoverageDimension(func::FuncOp kernel, uint64_t dimension,
                                        Value physicalExtent) {
  auto parameter = queryParameter(physicalExtent);
  PhysicalExprAttr parameterExtent = queryLaunchExpression(physicalExtent);
  if (!parameterExtent ||
      (!parameter && parameterExtent.getKind() !=
                         PhysicalExprKind::Constant))
    return failure();
  std::function<bool(PhysicalExprAttr)> hasBlockedExtent =
      [&](PhysicalExprAttr extent) {
    if (extent.getKind() == PhysicalExprKind::Parameter) {
      auto declaration = queryParameterBySymbol(kernel, extent.getParameterReference().getName());
      return succeeded(declaration) && *declaration != parameter &&
             declaration->getRole() !=
                 ParameterRole::FullCoverage;
    }
    return llvm::any_of(extent.getOperands(), [&](Attribute operand) {
      return hasBlockedExtent(cast<PhysicalExprAttr>(operand));
    });
  };
  SmallVector<MakeRangeOp> ranges;
  kernel.walk([&](MakeRangeOp range) {
    FailureOr<int64_t> sourceDimension = querySourceDimension(
        range.getResult().getType(), sourceAxisIdentity(range));
    auto fragment = dyn_cast<FragmentType>(range.getResult().getType());
    if (failed(sourceDimension) ||
        *sourceDimension != static_cast<int64_t>(dimension) ||
        !fragment ||
        fragment.getShape().size() != 1)
      return;
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
    if (hasBlockedExtent(extent))
      return;
    ranges.push_back(range);
  });
  // Validity can carry an independent occurrence of the same coordinates.
  SmallVector<MakeRangeOp> authorities(ranges.begin(), ranges.end());
  kernel.walk([&](MakeRangeOp range) {
    if (llvm::is_contained(ranges, range) ||
        range->getParentOfType<RegionFoldOp>() ||
        range->getParentOfType<RegionScanOp>())
      return;
    auto sourceDimension = queryRangeDimension(range);
    if (failed(sourceDimension) ||
        *sourceDimension != static_cast<int64_t>(dimension))
      return;
    if (llvm::any_of(authorities, [&](MakeRangeOp authority) {
          return sameLogicalRange(range, authority) &&
                 samePhysicalScalarExpression(range.getStart(),
                                              authority.getStart());
        }))
      ranges.push_back(range);
  });
  bool alreadyBound = !ranges.empty() &&
                      llvm::all_of(ranges, [&](MakeRangeOp range) {
                        return range.getExtent() == physicalExtent;
                      });
  for (MakeRangeOp range : ranges)
    if (failed(retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                                    parameterExtent,
                                    static_cast<int64_t>(dimension))))
      return failure();
  if (ranges.empty() || alreadyBound)
    return success();

  llvm::SmallPtrSet<Operation *, 32> rangeSet;
  for (MakeRangeOp range : ranges)
    rangeSet.insert(range.getOperation());

  SmallVector<AccessOpInterface> accesses;
  kernel.walk([&](AccessOpInterface access) { accesses.push_back(access); });
  // Preserve the existing read-before-write schema update order, including
  // operation order within each semantic kind. All footprints are read before
  // any range or access operand is changed below.
  llvm::stable_sort(accesses, [](AccessOpInterface lhs, AccessOpInterface rhs) {
    return lhs.getAccessKind() < rhs.getAccessKind();
  });
  llvm::DenseMap<Operation *, SmallVector<MakeRangeOp>> accessRanges;
  PhysicalProgramAnalysis accessAnalysis(kernel);
  auto recordAccessRanges = [&](Operation *access) -> LogicalResult {
    PhysicalAccessFootprint footprint = accessAnalysis.footprint(access);
    if (footprint.state != PhysicalFactState::Exact ||
        footprint.rangeState != PhysicalFactState::Exact)
      return access->emitOpError(
          "full-coverage rewrite requires one exact physical access footprint");
    auto &relevant = accessRanges[access];
    for (MakeRangeOp range : footprint.ranges)
      if (rangeSet.contains(range.getOperation()) &&
          !llvm::is_contained(relevant, range))
        relevant.push_back(range);
    return success();
  };
  for (AccessOpInterface access : accesses)
    if (failed(recordAccessRanges(access)))
      return failure();

  llvm::DenseMap<Operation *, Value> predicates;
  for (MakeRangeOp range : ranges) {
    OpBuilder builder(range);
    range->setOperand(1, physicalExtent);
    builder.setInsertionPointAfter(range);
    Value distance = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), range.getLogicalStop(),
        range.getLogicalStart(), BinaryOperator::Subtract);
    Value one = builder.create<arith::ConstantIndexOp>(range.getLoc(), 1);
    Value adjusted = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), distance,
        builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                 range.getStep(), one,
                                 BinaryOperator::Subtract),
        BinaryOperator::Add);
    Value logicalExtent = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), adjusted, range.getStep(),
        BinaryOperator::FloorDivide);
    Value logicalDistance = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), logicalExtent, range.getStep(),
        BinaryOperator::Multiply);
    auto coordinate = cast<FragmentType>(range.getResult().getType());
    Value member = range.getResult();
    Value stop;
    if (isUnitStepRange(range) &&
        (!queryNonNegativeIndexUpperBound(range.getLogicalStart()) ||
         !queryNonNegativeIndexUpperBound(range.getLogicalStop()))) {
      Value nonempty = builder.create<CompareOp>(
          range.getLoc(), builder.getI1Type(), range.getLogicalStart(),
          range.getLogicalStop(), ComparePredicate::Lt);
      Value zero = builder.create<arith::ConstantIndexOp>(range.getLoc(), 0);
      stop = builder.create<SelectOp>(range.getLoc(), builder.getIndexType(),
                                       nonempty, logicalDistance, zero);
      Value base = builder.create<BroadcastOp>(range.getLoc(), coordinate,
                                               range.getStart());
      // Unit-step offsets remain [0, extent) even when an inactive padded
      // absolute coordinate wraps. Preserve the range's physical origin.
      member = builder.create<BinaryOp>(range.getLoc(), coordinate, member,
                                         base, BinaryOperator::Subtract);
    } else {
      stop = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), range.getStart(), logicalDistance,
          BinaryOperator::Add);
    }
    Value stopFragment =
        builder.create<BroadcastOp>(range.getLoc(), coordinate, stop);
    auto predicate = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), coordinate.getShape(),
        coordinate.getAxisMaps(), coordinate.getValidity(),
        coordinate.getOwner());
    auto tail = builder.create<CompareOp>(range.getLoc(), predicate,
                                          member, stopFragment,
                                          ComparePredicate::Lt);
    tail->setAttr(physicalTailAttr, builder.getUnitAttr());
    predicates[range.getOperation()] = tail.getResult();
  }

  auto materializeTail = [&](OpBuilder &builder, Location location,
                             FragmentType target,
                             ArrayRef<MakeRangeOp> sources,
                             Value data = Value()) -> FailureOr<Value> {
    Value result;
    for (MakeRangeOp range : sources) {
      SmallVector<unsigned, 2> axes;
      for (PhysicalAxisProjection projection : queryRangeProjections(target, range))
        axes.push_back(projection.fragmentAxis);
      if (axes.empty() && data) {
        PhysicalRangeAxisFact relation =
            PhysicalProgramAnalysis(kernel).rangeAxes(data, {range});
        if (relation.isExact())
          axes.append(relation.fragmentAxes.begin(), relation.fragmentAxes.end());
      }
      if (axes.empty())
        return failure();
      for (unsigned axis : axes) {
        FailureOr<Value> current = projectPredicateToFragmentAxis(
            builder, location, predicates.lookup(range.getOperation()), target,
            axis);
        if (failed(current))
          return failure();
        result = result ? Value(builder.create<BinaryOp>(
                              location, current->getType(), result, *current,
                              BinaryOperator::LogicalAnd))
                        : *current;
      }
    }
    return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
  };

  for (AccessOpInterface access : accesses) {
    auto sources = accessRanges.lookup(access.getOperation());
    auto type = dyn_cast<FragmentType>(access.getAccessValueType());
    if (!access->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(access);
    SmallVector<Value> payloads(access.getAccessPayloads());
    if (access.getAccessKind() == AccessKind::Store) {
      SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
      for (MakeRangeOp range : sources) {
        auto coordinate = cast<FragmentType>(range.getResult().getType());
        for (PhysicalAxisProjection projection : queryRangeProjections(type, range))
          shape[projection.fragmentAxis] = coordinate.getShape()[0];
      }
      type = FragmentType::get(kernel.getContext(), type.getElementType(),
                               builder.getArrayAttr(shape), type.getAxisMaps(),
                               type.getValidity(), type.getOwner());
      // Only ordinary stores project their payload to the address-owned extent.
      // Atomic and scatter payloads retain their original schemas and semantics.
      auto value = projectPhysicalValueToSchema(builder, access.getLoc(), payloads.front(), type);
      if (failed(value))
        return access.emitOpError("full-coverage stored value has no exact physical projection");
      payloads.front() = *value;
    }
    Value data = access.getAccessKind() == AccessKind::Load ? access.getAccessResult() : Value();
    auto tail = materializeTail(builder, access.getLoc(), type, sources, data);
    if (failed(tail))
      return access.emitOpError("cannot project full-coverage dimension to access validity")
             << "; dimension=" << dimension << "; value=" << type
             << "; range_count=" << sources.size();
    Value valid = *tail;
    auto predicate = cast<FragmentType>(valid.getType());
    if (Value existing = access.getAccessValidity()) {
      if (existing.getType() != predicate) {
        auto projected = projectPhysicalValueToSchema(builder, access.getLoc(), existing, predicate);
        if (failed(projected))
          return access.emitOpError("full-coverage validity has no exact physical projection")
                 << "; existing=" << existing.getType() << "; required=" << predicate;
        existing = *projected;
      }
      valid = builder.create<BinaryOp>(access.getLoc(), predicate, existing, valid,
                                       BinaryOperator::LogicalAnd);
    }
    Value fill = access.getAccessFill();
    if (access.getAccessFillMutable()) {
      auto projected = fill ? projectPhysicalValueToSchema(builder, access.getLoc(), fill, type)
                            : materializeZeroFragment(builder, access.getLoc(), type);
      if (failed(projected))
        return access.emitOpError("full-coverage fill has no exact physical projection");
      fill = *projected;
    }
    // Preserve the access result identity: callers may be realizing coverage
    // for this exact SSA value. No operation or effect is cloned here.
    if (failed(access.updateAccessOperands(access.getAccessCoordinates(), payloads, valid, fill)))
      return failure();
  }
  return success();
}

} // namespace intent::gpu
