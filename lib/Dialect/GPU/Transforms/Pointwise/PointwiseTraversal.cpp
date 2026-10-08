#include "Pointwise.h"
#include "../Value/ReplayPolicy.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/RegionUtils.h"
#include <algorithm>

using namespace mlir;

namespace intent::gpu::pointwise {

Type replaceTraversalExtent(Type type, PhysicalSourceAxis source,
                            ArrayRef<int64_t> dimensions,
                            PhysicalExprAttr extent) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    SmallVector<Attribute> shape(fragment.getShape().begin(),
                                 fragment.getShape().end());
    bool changed = false;
    for (const PhysicalAxisProjection &projection :
         queryFragmentAxes(fragment, source)) {
      if (!llvm::is_contained(dimensions, projection.dimensionId))
        continue;
      shape[projection.fragmentAxis] = extent;
      changed = true;
    }
    return changed ? Type(FragmentType::get(
                         fragment.getContext(), fragment.getElementType(),
                         ArrayAttr::get(fragment.getContext(), shape),
                         fragment.getAxisMaps(), fragment.getValidity(),
                         fragment.getOwner()))
                   : type;
  }
  auto record = dyn_cast<RecordType>(type);
  if (!record)
    return type;
  SmallVector<Attribute> fields;
  bool changed = false;
  for (Attribute field : record.getFieldTypes()) {
    Type current = cast<TypeAttr>(field).getValue();
    Type replacement =
        replaceTraversalExtent(current, source, dimensions, extent);
    fields.push_back(TypeAttr::get(replacement));
    changed |= replacement != current;
  }
  return changed ? Type(RecordType::get(
                       type.getContext(), record.getFieldNames(),
                       ArrayAttr::get(type.getContext(), fields),
                       record.getOwner()))
                 : type;
}

FailureOr<Value> replayPointwiseValue(OpBuilder &builder, Value value,
                                      PhysicalSourceAxis source,
                                      ArrayRef<int64_t> traversalDimensions,
                                      PhysicalExprAttr blockedExtent,
                                      Value blockedRange, Value blockedValidity,
                                      Operation *insertionAnchor,
                                      IRMapping &mapping,
                                      const ReplayPolicy *policy) {
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  std::optional<ReplayPolicy> localPolicy;
  if (!policy) {
    localPolicy.emplace(kernel, ValueRange{value}, ArrayRef<Operation *>{insertionAnchor},
                        [&](Value current) { return containsTraversal(current, source, traversalDimensions); });
    policy = &*localPolicy;
  }
  for (int64_t dimension : traversalDimensions)
    if (failed(policy->bindSlices(builder, value, source, dimension, blockedExtent,
                                  blockedRange, insertionAnchor, mapping, {},
                                  traversalDimensions)))
      return failure();
  if (Value replacement = mapping.lookupOrNull(value)) return replacement;
  auto blocked = blockedRange.getDefiningOp<MakeRangeOp>();
  FailureOr<int64_t> blockedDimension =
      blocked ? queryRangeDimension(blocked) : FailureOr<int64_t>(failure());
  if (failed(blockedDimension))
    return insertionAnchor->emitOpError(
        "pointwise replay has no exact blocked-dimension authority");
  if (!containsTraversal(value, source, traversalDimensions)) {
    DominanceInfo dominance(kernel);
    if (dominance.dominates(value, insertionAnchor))
      return value;
  }
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return insertionAnchor->emitOpError(
        "pointwise replay cannot rematerialize a non-dominating block argument");
  auto selectedResult = dyn_cast<OpResult>(value);
  if (!selectedResult ||
      selectedResult.getResultNumber() >= producer->getNumResults())
    return producer->emitOpError(
        "pointwise replay has no exact producer result occurrence");
  if (auto original = dyn_cast<FragmentType>(value.getType());
      original && isa<ReshapeOp, scf::ForOp>(producer)) {
    auto target = cast<FragmentType>(replaceTraversalExtent(
        original, source, traversalDimensions, blockedExtent));
    SmallVector<unsigned> changedAxes;
    for (auto [axis, extent] : llvm::enumerate(original.getShape()))
      if (extent != target.getShape()[axis])
        changedAxes.push_back(axis);
    if (changedAxes.size() == 1) {
      unsigned axis = changedAxes.front();
      bool retain = false;
      if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
        auto relations = queryFragmentOperandRelations(reshape.getOperation());
        if (succeeded(relations))
          if (const auto *group = relations->front().groupForResultAxis(axis))
            retain = group->resultAxes.size() > 1;
      } else {
        auto relation = cast<AxisMapAttr>(original.getAxisMaps()[axis]);
        auto dependency = PhysicalProgramAnalysis(kernel).reductionDependency(
            value, sourceAxisIdentity(relation), relation.getDimensionId());
        retain = !dependency.isExact() || dependency.depends;
      }
      auto extent = cast<PhysicalExprAttr>(original.getShape()[axis]);
      auto size = constantLogicalRangeCardinality(blocked);
      auto begin = queryLaunchExpression(blocked.getLogicalStart());
      if (retain && size && isUnitStepRange(blocked) &&
          extent.getKind() == PhysicalExprKind::Constant &&
          extent.getValue() == *size && begin &&
          begin.getKind() == PhysicalExprKind::Constant &&
          begin.getValue() == 0 && DominanceInfo(kernel).dominates(value, insertionAnchor)) {
        // Preserve split shapes and complete loop-carried reduction inputs.
        // Only the already-computed result is projected to the writeback tile.
        Location location = producer->getLoc();
        auto axisMap = cast<AxisMapAttr>(original.getAxisMaps()[axis]);
        auto coordinate = FragmentType::get(
            kernel.getContext(), builder.getIndexType(), builder.getArrayAttr({blockedExtent}),
            builder.getArrayAttr({AxisMapAttr::get(
                kernel.getContext(), axisMap.getSourceId(), axisMap.getSourceAxis(),
                axisMap.getDimensionId(), 0, axisMap.getDerived())}),
            original.getValidity(), original.getOwner());
        auto identity = builder.getArrayAttr({ReshapeGroupAttr::get(
            kernel.getContext(), builder.getDenseI64ArrayAttr({0}),
            builder.getDenseI64ArrayAttr({0}))});
        Value indices = builder.create<ReshapeOp>(
            location, coordinate, blockedRange, identity);
        Value stop = builder.create<arith::ConstantIndexOp>(location, *size);
        Value valid = builder.create<CompareOp>(
            location, predicateType(coordinate), indices,
            builder.create<BroadcastOp>(location, coordinate, stop),
            ComparePredicate::Lt);
        Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
        Value nonnegative = builder.create<CompareOp>(
            location, predicateType(coordinate), indices,
            builder.create<BroadcastOp>(location, coordinate, zero),
            ComparePredicate::Ge);
        valid = builder.create<BinaryOp>(location, predicateType(coordinate),
                                          valid, nonnegative, BinaryOperator::LogicalAnd);
        auto projected = projectPredicateToFragmentAxis(builder, location, valid, target, axis);
        auto fill = materializeZeroFragment(builder, location, target);
        if (failed(projected) || failed(fill))
          return failure();
        auto selected = builder.create<GatherOp>(
            location, target, value, ValueRange{indices}, *projected, *fill,
            ArrayRef<int64_t>{static_cast<int64_t>(axis)});
        if (Attribute origin = producer->getAttr(originAttr))
          selected->setAttr(originAttr, origin);
        mapping.map(value, selected.getResult());
        return selected.getResult();
      }
    }
  }
  if (auto range = dyn_cast<MakeRangeOp>(producer)) {
    FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
    if (sourceAxisIdentity(range) == source &&
        sourceAxisIdentity(blocked) == source && succeeded(rangeDimension) &&
        *rangeDimension == *blockedDimension &&
        samePhysicalScalarExpression(range.getLogicalStart(), blocked.getLogicalStart()) &&
        samePhysicalScalarExpression(range.getStep(), blocked.getStep())) {
      mapping.map(value, blockedRange);
      return blockedRange;
    }
    auto originalType = dyn_cast<FragmentType>(range.getResult().getType());
    bool sameBounds =
        sourceAxisIdentity(range) == source &&
        samePhysicalScalarExpression(range.getLogicalStart(), blocked.getLogicalStart()) &&
        samePhysicalScalarExpression(range.getLogicalStop(), blocked.getLogicalStop()) &&
        samePhysicalScalarExpression(range.getStep(), blocked.getStep());
    auto sourceSize = constantLogicalRangeCardinality(range);
    auto targetSize = constantLogicalRangeCardinality(blocked);
    bool positional = sourceAxisIdentity(range) == source &&
        succeeded(rangeDimension) &&
        llvm::is_contained(traversalDimensions, *rangeDimension) &&
        isUnitStepRange(range) && isUnitStepRange(blocked) &&
        sourceSize && targetSize && *sourceSize == *targetSize;
    if (!blocked || failed(rangeDimension) || failed(blockedDimension) ||
        (*rangeDimension != *blockedDimension && !sameBounds && !positional) || !originalType ||
        originalType.getShape().size() != 1) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "pointwise replay reached a range without an exact shared dimension relation");
      if (succeeded(rangeDimension))
        diagnostic << "; source_dimension=" << *rangeDimension;
      if (succeeded(blockedDimension))
        diagnostic << "; blocked_dimension=" << *blockedDimension;
      return failure();
    }
    auto projectedType = FragmentType::get(
        originalType.getContext(), originalType.getElementType(),
        ArrayAttr::get(originalType.getContext(), {blockedExtent}),
        originalType.getAxisMaps(), originalType.getValidity(),
        originalType.getOwner());
    Value offset = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), blocked.getStart(),
        blocked.getLogicalStart(), BinaryOperator::Subtract);
    Value start = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), range.getLogicalStart(), offset,
        BinaryOperator::Add);
    Value projected = builder.create<MakeRangeOp>(
        range.getLoc(), projectedType, start, blocked.getExtent(),
        blocked.getStep(), range.getLogicalStart(), range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    inheritRangeAuthority(projected.getDefiningOp(), range);
    mapping.map(value, projected);
    return projected;
  }
  SmallVector<int64_t> valueDimensions;
  collectTraversalDimensions(value.getType(), source, valueDimensions);
  for (int64_t dimension : valueDimensions) {
    if (!llvm::is_contained(traversalDimensions, dimension))
      continue;
    PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayAt(
        value, source, PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/true, insertionAnchor, mapping, dimension);
    if (!replay.isReplayable()) {
      InFlightDiagnostic diagnostic = producer->emitOpError(
          "pointwise value has no exact insertion-point replay fact");
      diagnostic << "; dimension=" << dimension;
      for (Operation *blocker : replay.blockers)
        diagnostic << "; blocker=" << blocker->getName();
      return failure();
    }
  }
  if (auto contract = dyn_cast<ContractOp>(producer)) {
    auto requested = queryFragmentAxis(value.getType(), source);
    std::optional<unsigned> pairedAxis;
    unsigned resultAxis = 0;
    for (unsigned axis = 0;
         requested.isExact() &&
         axis < contract.getLhs().getType().getShape().size(); ++axis) {
      if (llvm::is_contained(contract.getLhsReductionAxes(),
                             static_cast<int64_t>(axis)))
        continue;
      if (resultAxis++ != requested.fragmentAxis)
        continue;
      auto pair = llvm::find(contract.getLhsBatchAxes(),
                             static_cast<int64_t>(axis));
      if (pair != contract.getLhsBatchAxes().end())
        pairedAxis = std::distance(contract.getLhsBatchAxes().begin(), pair);
      break;
    }
    if (requested.isExact() &&
        llvm::is_contained(traversalDimensions, requested.dimensionId) &&
        pairedAxis) {
      // A batch result selects the same ordinal from both operand axes, even
      // when their source identities differ. Preserve the selected tile's
      // coordinates and validity through each operand's existing replay path.
      unsigned batch = *pairedAxis;
      SmallVector<unsigned> axes{
          static_cast<unsigned>(contract.getLhsBatchAxes()[batch]),
          static_cast<unsigned>(contract.getRhsBatchAxes()[batch]),
          requested.fragmentAxis};
      SmallVector<Value> operands;
      for (auto [operand, axis] :
           llvm::zip(contract->getOperands(), axes)) {
        auto type = cast<FragmentType>(operand.getType());
        auto relation = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
        auto inputSource = sourceAxisIdentity(relation);
        auto inputAxis = queryFragmentAxis(type, inputSource,
                                           relation.getDimensionId());
        if (!inputAxis.isExact() || inputAxis.fragmentAxis != axis)
          return contract.emitOpError(
              "batch replay has no unique operand-axis relation");
        SmallVector<int64_t> dimensions{relation.getDimensionId()};
        // The same SSA value can occupy different paired roles. A replacement
        // computed for one operand must not become the other operand's input.
        IRMapping operandMapping;
        auto replayed = replayPointwiseValue(
            builder, operand, inputSource, dimensions, blockedExtent,
            blockedRange, blockedValidity, insertionAnchor, operandMapping, policy);
        if (failed(replayed))
          return failure();
        auto expected = replaceTraversalExtent(
            type, inputSource, dimensions, blockedExtent);
        if ((*replayed).getType() != expected)
          return contract.emitOpError(
              "batch replay did not preserve the paired tile schema");
        operands.push_back(*replayed);
      }
      IRMapping cloneMapping;
      auto clone = cast<ContractOp>(builder.clone(*producer, cloneMapping));
      clone->setOperands(operands);
      clone.getResult().setType(cast<FragmentType>(replaceTraversalExtent(
          value.getType(), source, traversalDimensions, blockedExtent)));
      mapping.map(value, clone.getResult());
      return clone.getResult();
    }
  }
  SmallVector<Value> replayOperands(producer->getOperands());
  if (isa<scf::IfOp, scf::ForOp>(producer)) {
    llvm::SetVector<Value> captures;
    for (Region &region : producer->getRegions())
      getUsedValuesDefinedAbove(region, captures);
    replayOperands.append(captures.begin(), captures.end());
  }
  auto projectionRelations = queryFragmentOperandRelations(selectedResult);
  for (auto [operandNumber, operand] : llvm::enumerate(replayOperands)) {
    PhysicalSourceAxis operandSource = source;
    SmallVector<int64_t> operandDimensions(traversalDimensions);
    if (isa<BroadcastOp, ReshapeOp>(producer)) {
      auto input = dyn_cast<FragmentType>(operand.getType());
      auto output = dyn_cast<FragmentType>(value.getType());
      PhysicalAxisProjection requested = queryFragmentAxis(output, source);
      if (input && requested.isExact() &&
          llvm::is_contained(traversalDimensions, requested.dimensionId)) {
        std::optional<unsigned> inputAxis;
        if (succeeded(projectionRelations))
          for (const auto &relation : *projectionRelations) {
            if (relation.operandNumber != operandNumber)
              continue;
            const auto *group =
                relation.groupForResultAxis(requested.fragmentAxis);
            if (!group || group->sourceAxes.size() != 1 ||
                group->resultAxes.size() != 1)
              continue;
            if (isa<BroadcastOp>(producer) ||
                (group->kind == FragmentAxisRelationKind::Reassociation &&
                 input.getShape()[group->sourceAxes[0]] ==
                     output.getShape()[requested.fragmentAxis]))
              inputAxis = group->sourceAxes[0];
          }
        if (auto axis = inputAxis) {
            auto axisMap = cast<AxisMapAttr>(input.getAxisMaps()[*axis]);
            PhysicalRangeFact ranges =
                PhysicalProgramAnalysis(kernel).axisRanges(operand, *axis);
            bool renamedAxis =
                input.getShape()[*axis] ==
                    output.getShape()[requested.fragmentAxis];
            if (renamedAxis ||
                (axisMap.getDimensionId() == *blockedDimension &&
                ranges.state != PhysicalFactState::Unknown &&
                ranges.blockers.empty() &&
                PhysicalProgramAnalysis(kernel).lockstepRanges(ranges.roots).isExact() &&
                llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
                  auto dimension = queryRangeDimension(range);
                  bool sameExtent = succeeded(dimension) &&
                                    *dimension == *blockedDimension;
                  bool sameBounds = samePhysicalScalarExpression(
                      range.getLogicalStart(), blocked.getLogicalStart()) &&
                      samePhysicalScalarExpression(range.getLogicalStop(), blocked.getLogicalStop());
                  auto sourceSize = constantLogicalRangeCardinality(range);
                  auto targetSize = constantLogicalRangeCardinality(blocked);
                  bool sameCardinality = renamedAxis && sourceSize && targetSize &&
                                         *sourceSize == *targetSize;
                  return (sameExtent || sameBounds || sameCardinality) &&
                         samePhysicalScalarExpression(range.getStep(), blocked.getStep());
                }))) {
              // An explicit projection can rename an exact coordinate traversal.
              operandSource = sourceAxisIdentity(axisMap);
              for (int64_t &dimension : operandDimensions)
                if (dimension == requested.dimensionId)
                  dimension = axisMap.getDimensionId();
            }
        }
      }
    }
    FailureOr<Value> replacement = replayPointwiseValue(
        builder, operand, operandSource, operandDimensions, blockedExtent,
        blockedRange, blockedValidity, insertionAnchor, mapping, policy);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
  }
  auto result = dyn_cast<FragmentType>(value.getType());
  FragmentType resultType =
      result ? dyn_cast<FragmentType>(replaceTraversalExtent(
                   result, source, traversalDimensions, blockedExtent))
             : FragmentType();
  auto mapped = [&](Value operand) {
    if (!operand)
      return Value();
    Value replacement = mapping.lookupOrNull(operand);
    return replacement ? replacement : operand;
  };
  auto accessValidity = [&](Value existing) -> FailureOr<Value> {
    Value valid = mapped(existing);
    for (Value operand : producer->getOperands()) {
      Value coordinate = mapped(operand);
      if (coordinate == operand || !coordinate.getDefiningOp<MakeRangeOp>())
        continue;
      auto type = cast<FragmentType>(coordinate.getType());
      Value zero = builder.create<arith::ConstantIndexOp>(producer->getLoc(), 0);
      Value lower = builder.create<CompareOp>(
          producer->getLoc(), predicateType(type), coordinate,
          builder.create<SplatOp>(producer->getLoc(), type, zero), ComparePredicate::Ge);
      FailureOr<Value> bounded = materializeValidityConjunction(
          builder, producer->getLoc(), valid, lower, predicateType(resultType));
      if (failed(bounded))
        return failure();
      valid = *bounded;
    }
    if (!blockedValidity)
      return valid;
    auto axes = queryFragmentAxes(resultType, source);
    llvm::erase_if(axes, [&](const PhysicalAxisProjection &axis) {
      return !llvm::is_contained(traversalDimensions, axis.dimensionId);
    });
    FailureOr<Value> projected = axes.size() == 1
        ? projectPredicateToFragmentAxis(builder, producer->getLoc(),
                                         blockedValidity, resultType,
                                         axes.front().fragmentAxis)
        : projectPhysicalValueToSchema(builder, producer->getLoc(),
                                          blockedValidity, predicateType(resultType));
    if (failed(projected))
      return producer->emitOpError("pointwise tile validity lost its access-axis projection");
    if (!valid)
      return *projected;
    FailureOr<Value> original =
        projectPhysicalValueToSchema(builder, producer->getLoc(), valid, predicateType(resultType));
    if (failed(original))
      return failure();
    return Value(builder.create<BinaryOp>(producer->getLoc(),
                                          predicateType(resultType), *original,
                                          *projected,
                                          BinaryOperator::LogicalAnd));
  };
  Value replayed;
  if (auto load = dyn_cast<LoadOp>(producer)) {
    if (!resultType)
      return failure();
    FailureOr<Value> valid = accessValidity(load.getValid());
    if (failed(valid))
      return failure();
    Value fill = mapped(load.getFill());
    if (*valid && !fill) {
      FailureOr<Value> zero = materializeZeroFragment(builder, producer->getLoc(), resultType);
      if (failed(zero))
        return failure();
      fill = *zero;
    }
    SmallVector<Value> coordinates;
    for (Value coordinate : load.getCoordinates())
      coordinates.push_back(mapped(coordinate));
    auto clone = builder.create<LoadOp>(
        producer->getLoc(), resultType, mapped(load.getResource()), coordinates,
        *valid, fill, load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      clone->setAttr(originAttr, origin);
    replayed = clone.getResult();
  } else if (auto gather = dyn_cast<GatherOp>(producer)) {
    if (!resultType)
      return failure();
    FailureOr<Value> valid = accessValidity(gather.getValid());
    if (failed(valid))
      return failure();
    Value fill = mapped(gather.getFill());
    if (*valid && !fill) {
      FailureOr<Value> zero = materializeZeroFragment(builder, producer->getLoc(), resultType);
      if (failed(zero))
        return failure();
      fill = *zero;
    }
    SmallVector<Value> coordinates;
    for (Value coordinate : gather.getCoordinates())
      coordinates.push_back(mapped(coordinate));
    auto clone = builder.create<GatherOp>(
        producer->getLoc(), resultType, mapped(gather.getSource()), coordinates,
        *valid, fill, gather.getSourceAxes());
    if (Attribute origin = gather->getAttr(originAttr))
      clone->setAttr(originAttr, origin);
    replayed = clone.getResult();
  } else if (auto reshape = dyn_cast<ReshapeOp>(producer);
             reshape && !containsSource(reshape.getValue(), source) &&
             mapped(reshape.getValue()) == reshape.getValue()) {
    if (!resultType)
      return failure();
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, producer->getLoc(), value, resultType);
    if (failed(projected))
      return producer->emitOpError("retained reshape cannot adopt the pointwise tile")
             << "; value=" << value.getType() << "; target=" << resultType;
    replayed = *projected;
  } else {
    SmallVector<MakeRangeOp> controlSourceRanges;
    if (isa<scf::IfOp, scf::ForOp>(producer) && result) {
      PhysicalProgramAnalysis analysis(kernel);
      for (auto [axis, attribute] : llvm::enumerate(result.getAxisMaps())) {
        auto relation = cast<AxisMapAttr>(attribute);
        if (!(sourceAxisIdentity(relation) == source) ||
            !llvm::is_contained(traversalDimensions, relation.getDimensionId()))
          continue;
        PhysicalRangeFact ranges = analysis.axisRanges(value, axis);
        if (!ranges.isExact())
          continue;
        for (MakeRangeOp range : ranges.roots) {
          auto sourceSize = constantLogicalRangeCardinality(range);
          auto targetSize = constantLogicalRangeCardinality(blocked);
          if (producer->isProperAncestor(range) && sourceSize && targetSize &&
              *sourceSize == *targetSize &&
              samePhysicalScalarExpression(range.getStep(), blocked.getStep()) &&
              !llvm::is_contained(controlSourceRanges, range))
            controlSourceRanges.push_back(range);
        }
      }
    }
    auto cloned = cloneWithPhysicalSchema(builder, producer, mapping,
        [&](Value original) {
          return replaceTraversalExtent(original.getType(), source,
                                         traversalDimensions, blockedExtent);
        });
    if (failed(cloned))
      return producer->emitOpError(
          "pointwise replay cannot transport the producer's selected schema")
          << "; result=" << value.getType()
          << "; selected=" << replaceTraversalExtent(
                 value.getType(), source, traversalDimensions, blockedExtent);
    replayed = (*cloned)[selectedResult.getResultNumber()];
    bool structuredControl = isa<scf::IfOp, scf::ForOp>(producer);
    llvm::SmallPtrSet<Operation *, 8> controlRanges;
    for (MakeRangeOp sourceRange : controlSourceRanges)
      if (Value mappedRange = mapping.lookupOrNull(sourceRange.getResult()))
        if (auto range = mappedRange.getDefiningOp<MakeRangeOp>())
          controlRanges.insert(range);
    llvm::DenseMap<Value, Value> controlTails;
    if (isa<ReduceOp, ScanOp, RegionFoldOp, RegionScanOp>(producer) || structuredControl)
      for (Region &region : replayed.getDefiningOp()->getRegions())
        for (Block &block : region) {
          WalkResult walked = block.walk([&](Operation *nested) -> WalkResult {
            auto range = dyn_cast<MakeRangeOp>(nested);
            FailureOr<int64_t> dimension =
                range ? queryRangeDimension(range)
                      : FailureOr<int64_t>(failure());
            if (!range ||
                (!controlRanges.contains(range) &&
                 (!(sourceAxisIdentity(range) == source) || failed(dimension) ||
                  !llvm::is_contained(traversalDimensions, *dimension))))
              return WalkResult::advance();
            if (structuredControl) {
              if (failed(retargetSourceExtent(range.getResult(),
                                              sourceAxisIdentity(range),
                                              blockedExtent)))
                return WalkResult::interrupt();
              OpBuilder nestedBuilder(range);
              Value offset = nestedBuilder.create<BinaryOp>(range.getLoc(), nestedBuilder.getIndexType(),
                  blocked.getStart(), blocked.getLogicalStart(), BinaryOperator::Subtract);
              Value start = nestedBuilder.create<BinaryOp>(range.getLoc(), nestedBuilder.getIndexType(),
                  range.getLogicalStart(), offset, BinaryOperator::Add);
              range->setOperand(0, start);
              range->setOperand(1, blocked.getExtent());
              nestedBuilder.setInsertionPointAfter(range);
              auto type = range.getResult().getType();
              Value zero = nestedBuilder.create<arith::ConstantIndexOp>(range.getLoc(), 0);
              Value lower = nestedBuilder.create<CompareOp>(range.getLoc(), predicateType(type),
                  range, nestedBuilder.create<SplatOp>(range.getLoc(), type, zero), ComparePredicate::Ge);
              Value upper = nestedBuilder.create<CompareOp>(range.getLoc(), predicateType(type),
                  range, nestedBuilder.create<SplatOp>(range.getLoc(), type, range.getLogicalStop()), ComparePredicate::Lt);
              controlTails[range] = nestedBuilder.create<BinaryOp>(range.getLoc(), predicateType(type),
                  lower, upper, BinaryOperator::LogicalAnd);
              return WalkResult::advance();
            }
            // Helper-local ranges are complete physical values too.  When a
            // structured producer is replayed for a wider ownership fragment,
            // its local coordinate carrier must use that same physical extent;
            // changing only the result type leaves an illegal one-lane range.
            bool coveredItsLocalDomain =
                samePhysicalScalarExpression(range.getStart(),
                                             range.getLogicalStart()) &&
                samePhysicalScalarExpression(range.getExtent(),
                                             range.getLogicalStop());
            range->setOperand(1, blocked.getExtent());
            if (coveredItsLocalDomain)
              range->setOperand(4, blocked.getExtent());
            return WalkResult::advance();
          });
          if (walked.wasInterrupted())
            return failure();
        }
    if (!controlTails.empty() && failed(addTailValidity(kernel, kernel, controlTails,
                                                       /*includeStores=*/false)))
      return failure();
  }
  if (!mapping.lookupOrNull(value))
    mapping.map(value, replayed);
  return replayed;
}

LogicalResult realizeReusePointwiseTraversal(func::FuncOp kernel,
                                             MakeRangeOp range,
                                             ArrayRef<StoreOp> stores,
                                             bool effectLocal,
                                             ArrayRef<LoadOp> retainedReads,
                                             llvm::function_ref<void(StoreOp, StoreOp)> replaceStore) {
  IRMapping replayBindings;
  if (stores.empty())
    return range.emitOpError(
        "reuse-sensitive pointwise traversal has no write effect");
  FailureOr<SmallVector<int64_t>> traversalDimensions =
      traversalDimensionsForStores(stores, range);
  if (failed(traversalDimensions))
    return range.emitOpError(
        "reuse-sensitive pointwise traversal has no typed coordinate/data relation");
  bool accessDependentSubregion =
      !effectLocal && hasAccessDependentSubregionBounds(kernel, range);
  SmallVector<int64_t> payloadTraversalDimensions(*traversalDimensions);
  std::optional<int64_t> payloadTraversalDimension;
  if (accessDependentSubregion) {
    FailureOr<int64_t> dimension =
        reuseTraversalDimension(kernel, stores, range);
    if (failed(dimension))
      return range.emitOpError(
          "reuse-sensitive pointwise traversal has no unique reduced output dimension");
    payloadTraversalDimension = *dimension;
    payloadTraversalDimensions.assign(1, *dimension);
  }

  PhysicalSourceAxis logicalSource{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
  std::string name = ("POINTWISE_CHUNK_S" + Twine(logicalSource.sourceId) +
                      "_A" + Twine(logicalSource.sourceAxis))
                         .str();
  ParameterAttr chunk;
  bool ambiguousChunk = false;
  for (Attribute declaration : getParameterDeclarations(kernel)) {
    auto parameter = cast<ParameterAttr>(declaration);
    auto source = parameter.getBinding().getSource();
    if (!parameter.getBinding().getPointwiseChunk() || !source ||
        !(PhysicalSourceAxis{source.getSourceId(), source.getSourceAxis(),
                             source.getDerived()} == logicalSource))
      continue;
    if (chunk && chunk != parameter)
      ambiguousChunk = true;
    else
      chunk = parameter;
  }
  if (ambiguousChunk)
    return range.emitOpError(
        "pointwise traversal has multiple parameters for one source axis");
  if (!chunk) {
    SmallVector<int64_t> candidates{1,   2,   4,    8,    16,   32, 64,
                                   128, 256, 512, 1024, 2048, 4096};
    FailureOr<int64_t> staticExtent = exactStaticTraversalExtent(
        PhysicalProgramAnalysis(kernel).axisRanges(range.getResult(), 0));
    if (succeeded(staticExtent)) {
      uint64_t paddedExtent = llvm::PowerOf2Ceil(
          static_cast<uint64_t>(std::max<int64_t>(*staticExtent, 1)));
      llvm::erase_if(candidates, [&](int64_t candidate) {
        return static_cast<uint64_t>(candidate) > paddedExtent;
      });
    }
    uint32_t elementBitWidth = 0;
    for (StoreOp store : stores)
      elementBitWidth =
          std::max(elementBitWidth,
                   physicalElementBitWidth(store.getValue().getType()));
    if (elementBitWidth == 0)
      return range.emitOpError(
          "pointwise traversal has no typed data width for its physical parameter");
    OpBuilder entry(&kernel.getBody().front(), kernel.getBody().front().begin());
    auto schema = ParameterAttr::get(
        kernel.getContext(), entry.getStringAttr(name), entry.getIndexType(),
        ParameterRole::OwnershipN,
        ParameterCategory::Pointwise, elementBitWidth,
        DenseI64ArrayAttr::get(kernel.getContext(), candidates),
        ConfigurationBindingPhase::Shared,
        ParameterBindingAttr::get(kernel.getContext(), {},
            PhysicalSourceAttr::get(kernel.getContext(), logicalSource.sourceId,
                                   logicalSource.sourceAxis, logicalSource.derived),
            {}, {}, true, false));
    if (failed(declareParameter(kernel, schema))) return failure();
    chunk = schema;
  }

  auto originalType = dyn_cast<FragmentType>(range.getResult().getType());
  if (!originalType || originalType.getShape().size() != 1)
    return range.emitOpError(
        "reuse-sensitive pointwise traversal requires one physical source axis");
  auto boundedChunk = boundedTraversalChunk(chunk, range);
  if (failed(boundedChunk)) return failure();
  PhysicalExprAttr chunkExtent = *boundedChunk;
  SmallVector<Attribute> blockedMappings(originalType.getAxisMaps().begin(),
                                         originalType.getAxisMaps().end());
  if (accessDependentSubregion) {
    auto mapping = cast<AxisMapAttr>(blockedMappings[0]);
    blockedMappings[0] = AxisMapAttr::get(
        kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        *payloadTraversalDimension, 0, mapping.getDerived());
  }
  auto blockedType = FragmentType::get(
      kernel.getContext(), originalType.getElementType(),
      ArrayAttr::get(kernel.getContext(), {chunkExtent}),
      ArrayAttr::get(kernel.getContext(), blockedMappings),
      originalType.getValidity(),
      originalType.getOwner());
  if (!effectLocal && retainedReads.empty()) {
    auto retained = retainCoveredPointwiseGraph(kernel, range, stores);
    if (failed(retained)) return failure();
  }
  // The original range remains the authority for the full reduction
  // traversal.  The internal writeback loop below owns a distinct blocked
  // range and replays only the store-side value graph against it.

  llvm::MapVector<Block *, SmallVector<StoreOp>> storesByBlock;
  for (StoreOp store : stores)
    storesByBlock[store->getBlock()].push_back(store);
  SmallVector<SmallVector<StoreOp>> storeGroups;
  PhysicalProgramAnalysis replayAnalysis(kernel);
  auto canReplayAt = [&](StoreOp store, Operation *anchor) {
    DominanceInfo dominance(kernel);
    for (LoadOp read : retainedReads)
      if (dominance.dominates(read.getResult(), store) &&
          !dominance.dominates(read.getResult(), anchor))
        return false;
    auto replayable = [&](Value value, ArrayRef<int64_t> dimensions) {
      if (!value)
        return true;
      SmallVector<int64_t> valueDimensions;
      collectTraversalDimensions(value.getType(), logicalSource,
                                 valueDimensions);
      for (int64_t dimension : valueDimensions)
        if (llvm::is_contained(dimensions, dimension) &&
            !replayAnalysis
                 .replayAt(value, logicalSource,
                                PhysicalReplayScope::ValueGraph,
                                /*allowAccesses=*/true, anchor, replayBindings,
                                dimension)
                 .isReplayable())
          return false;
      return true;
    };
    if (!replayable(store.getValue(), payloadTraversalDimensions) ||
        !replayable(store.getValid(), payloadTraversalDimensions))
      return false;
    return llvm::all_of(store.getCoordinates(), [&](Value coordinate) {
      return replayable(coordinate, *traversalDimensions);
    });
  };
  for (auto &entry : storesByBlock) {
    SmallVector<StoreOp> &blockStores = entry.second;
    SmallVector<bool> assigned(blockStores.size(), false);
    for (unsigned first = 0; first < blockStores.size(); ++first) {
      if (assigned[first])
        continue;
      SmallVector<StoreOp> group{blockStores[first]};
      assigned[first] = true;
      Operation *anchor = blockStores[first].getOperation();
      for (unsigned next = first + 1; next < blockStores.size(); ++next) {
        if (assigned[next] || !canReplayAt(blockStores[next], anchor))
          continue;
        group.push_back(blockStores[next]);
        assigned[next] = true;
      }
      storeGroups.push_back(std::move(group));
    }
  }
  SmallVector<scf::ForOp> materializedLoops;
  for (SmallVector<StoreOp> &group : storeGroups) {
  SmallVector<Value> replayRoots;
  SmallVector<Operation *> replacedConsumers;
  for (StoreOp store : group) {
    replayRoots.push_back(store.getValue());
    if (store.getValid()) replayRoots.push_back(store.getValid());
    llvm::append_range(replayRoots, store.getCoordinates());
    replacedConsumers.push_back(store);
  }
  ReplayPolicy replayPolicy(kernel, replayRoots, replacedConsumers,
      [&](Value current) {
        return containsTraversal(current, logicalSource, payloadTraversalDimensions) ||
               containsTraversal(current, logicalSource, *traversalDimensions);
      });
  OpBuilder builder(group.front());
  Operation *loopInsertionAnchor = group.front().getOperation();
  Value stop = range.getLogicalStop();
  Value chunkSize = builder.create<PhysicalExprOp>(
      range.getLoc(), builder.getIndexType(), chunkExtent);
  Value loopStep = builder.create<BinaryOp>(
      range.getLoc(), builder.getIndexType(), chunkSize, range.getStep(),
      BinaryOperator::Multiply);
  bool bodyFailed = false;
  std::string failureReason;
  auto loop = createTraversalLoop(builder,
      range.getLoc(), range.getStart(), stop, loopStep, ValueRange{},
      [&](OpBuilder &nested, Location location, Value tileStart, ValueRange) {
        Value blocked = nested.create<MakeRangeOp>(
            location, blockedType, tileStart, chunkSize, range.getStep(),
            range.getLogicalStart(), range.getLogicalStop(), range.getSourceId(),
            range.getSourceAxis(), range.getDerived());
        inheritRangeAuthority(blocked.getDefiningOp(), range);
        Value end = nested.create<BroadcastOp>(location, blockedType, stop);
        auto tailComparison = nested.create<CompareOp>(
            location, predicateType(blockedType), blocked, end,
            ComparePredicate::Lt);
        tailComparison->setAttr(physicalTailAttr, nested.getUnitAttr());
        Value tail = tailComparison.getResult();
        IRMapping mapping;
        mapping.map(range.getResult(), blocked);
        for (LoadOp read : retainedReads) {
          if (!DominanceInfo(kernel).dominates(read.getResult(), loopInsertionAnchor))
            continue;
          auto slice = materializeRetainedSlice(
              nested, location, read.getResult(), 0, chunkExtent, blocked,
              loopInsertionAnchor);
          if (failed(slice)) {
            bodyFailed = true;
            failureReason = "retained input cannot be sliced in the writeback loop";
            return;
          }
          mapping.map(read.getResult(), *slice);
        }
        for (StoreOp store : group) {
          FailureOr<Value> payload = replayPointwiseValue(
              nested, store.getValue(), logicalSource,
              payloadTraversalDimensions,
              chunkExtent,
              blocked, tail, loopInsertionAnchor, mapping, &replayPolicy);
          if (failed(payload)) {
            bodyFailed = true;
            failureReason = "write payload cannot be replayed in the internal tile loop";
            return;
          }
          SmallVector<Value> coordinates;
          for (Value coordinate : store.getCoordinates()) {
            FailureOr<Value> replayed = replayPointwiseValue(
                nested, coordinate, logicalSource, *traversalDimensions,
                chunkExtent, blocked,
                tail, loopInsertionAnchor, mapping, &replayPolicy);
            if (failed(replayed)) {
              bodyFailed = true;
              failureReason =
                  "write coordinate cannot be replayed in the internal tile loop";
              return;
            }
            coordinates.push_back(*replayed);
          }
          auto payloadType = dyn_cast<FragmentType>((*payload).getType());
          if (!payloadType) {
            FailureOr<FragmentType> schema =
                coordinateValueSchema((*payload).getType(), coordinates);
            if (failed(schema)) {
              bodyFailed = true;
              failureReason =
                  "write coordinates have no exact payload fragment schema";
              return;
            }
            FailureOr<Value> projected = projectPhysicalValueToSchema(
                nested, location, *payload, *schema);
            if (failed(projected)) {
              bodyFailed = true;
              failureReason =
                  "write payload cannot adopt its coordinate fragment schema";
              return;
            }
            payload = *projected;
            payloadType = *schema;
          }
          FailureOr<Value> valid = projectPhysicalValueToSchema(nested, location, tail,
                                               predicateType(payloadType));
          if (failed(valid)) {
            bodyFailed = true;
            failureReason = "tile tail cannot be projected to the write payload";
            return;
          }
          if (store.getValid()) {
            FailureOr<Value> existing = replayPointwiseValue(
                nested, store.getValid(), logicalSource,
                payloadTraversalDimensions,
                chunkExtent,
                blocked, tail, loopInsertionAnchor, mapping, &replayPolicy);
            if (failed(existing)) {
              bodyFailed = true;
              failureReason = "write validity cannot be replayed in the tile loop";
              return;
            }
            FailureOr<Value> projected = projectPhysicalValueToSchema(
                nested, location, *existing, predicateType(payloadType));
            if (failed(projected)) {
              bodyFailed = true;
              failureReason = "write validity cannot be projected to the payload";
              return;
            }
            valid = Value(nested.create<BinaryOp>(
                location, predicateType(payloadType), *valid, *projected,
                BinaryOperator::LogicalAnd));
          }
          auto replacement = nested.create<StoreOp>(
              location, store.getResource(), coordinates, *payload, *valid,
              store.getSourceAxes());
          if (Attribute origin = store->getAttr(originAttr))
            replacement->setAttr(originAttr, origin);
          replaceStore(store, replacement);
        }
        if (!bodyFailed)
          nested.create<scf::YieldOp>(location);
      });
  if (bodyFailed) {
    loop.erase();
    for (scf::ForOp materialized : materializedLoops)
      materialized.erase();
    return range.emitOpError(
               "reuse-sensitive pointwise traversal could not be materialized: ")
           << failureReason;
  }
  materializedLoops.push_back(loop);
  }
  for (StoreOp store : stores)
    store.erase();
  return success();
}

LogicalResult realizeOwnedHistograms(func::FuncOp kernel) {
  SmallVector<HistogramOp> histograms;
  kernel.walk([&](HistogramOp histogram) { histograms.push_back(histogram); });
  for (HistogramOp histogram : histograms) {
    SmallVector<StoreOp> stores;
    kernel.walk([&](StoreOp store) {
      if (histogramSource(store.getValue()) == histogram)
        stores.push_back(store);
    });
    if (stores.empty())
      return histogram.emitOpError(
          "histogram result has no physical output ownership effect");

    FailureOr<FragmentType> outputType = coordinateValueSchema(
        histogram.getResult().getType().getElementType(),
        stores.front().getCoordinates());
    if (failed(outputType) || outputType->getShape().size() != 1)
      return histogram.emitOpError(
          "histogram output has no one-axis physical ownership schema");
    for (StoreOp store : llvm::drop_begin(stores)) {
      FailureOr<FragmentType> current = coordinateValueSchema(
          histogram.getResult().getType().getElementType(),
          store.getCoordinates());
      if (failed(current) || *current != *outputType)
        return histogram.emitOpError(
            "histogram output effects do not share one physical ownership schema");
    }

    auto outputMapping = cast<AxisMapAttr>(outputType->getAxisMaps()[0]);
    PhysicalSourceAxis outputSource{outputMapping.getSourceId(),
                                    outputMapping.getSourceAxis(),
                                    outputMapping.getDerived()};
    llvm::SmallPtrSet<Operation *, 8> outputRoots;
    for (Value coordinate : stores.front().getCoordinates())
      collectCoordinateRanges(coordinate, outputRoots);
    SmallVector<MakeRangeOp> outputRanges;
    for (Operation *root : outputRoots)
      if (auto range = dyn_cast<MakeRangeOp>(root);
          range && sourceAxisIdentity(range) == outputSource)
        outputRanges.push_back(range);
    if (outputRanges.size() != 1 || !isUnitStepRange(outputRanges.front()))
      return histogram.emitOpError(
          "histogram output ownership has no unique unit-step bin range");
    MakeRangeOp outputRange = outputRanges.front();
    for (StoreOp store : llvm::drop_begin(stores)) {
      llvm::SmallPtrSet<Operation *, 8> currentRoots;
      for (Value coordinate : store.getCoordinates())
        collectCoordinateRanges(coordinate, currentRoots);
      SmallVector<MakeRangeOp> currentRanges;
      for (Operation *root : currentRoots)
        if (auto range = dyn_cast<MakeRangeOp>(root);
            range && sourceAxisIdentity(range) == outputSource)
          currentRanges.push_back(range);
      if (currentRanges.size() != 1 ||
          !PhysicalProgramAnalysis(kernel)
               .lockstepRanges({outputRange, currentRanges.front()})
               .isExact())
        return histogram.emitOpError(
            "histogram output effects do not share one physical bin traversal");
    }
    FailureOr<ParameterAttr> outputParameter =
        queryBlockingParameter(kernel, outputRange);
    if (failed(outputParameter))
      return histogram.emitOpError(
          "histogram output ownership has no typed blocking parameter");
    ParameterAttr outputSchema = *outputParameter;
    SmallVector<int64_t> outputCandidates(
        outputSchema.getCandidates().asArrayRef());
    if (auto bins = histogram.getBins().getDefiningOp<arith::ConstantIndexOp>())
      llvm::erase_if(outputCandidates,
                     [&](int64_t candidate) { return candidate > bins.value(); });
    if (outputCandidates.empty())
      return histogram.emitOpError(
          "histogram output ownership has no legal bin-tile candidate");
    if (failed(updateParameter(kernel, outputSchema.withCandidates(
            DenseI64ArrayAttr::get(kernel.getContext(), outputCandidates)))))
      return failure();

    auto valuesType = dyn_cast<FragmentType>(histogram.getValues().getType());
    if (!valuesType || valuesType.getShape().size() != 1)
      return histogram.emitOpError(
          "histogram input has no one-axis physical traversal schema");
    PhysicalRangeFact inputFact =
        PhysicalProgramAnalysis(kernel).axisRanges(histogram.getValues(), 0);
    if (!inputFact.isUnique() || !isUnitStepRange(inputFact.roots.front()))
      return histogram.emitOpError(
          "histogram input has no unique unit-step physical traversal");
    MakeRangeOp inputRange = inputFact.roots.front();
    FailureOr<int64_t> inputDimension = queryRangeDimension(inputRange);
    if (failed(inputDimension))
      return histogram.emitOpError(
          "histogram input traversal has no logical dimension identity");
    PhysicalSourceAxis inputSource = sourceAxisIdentity(inputRange);

    std::string chunkName =
        ("HISTOGRAM_CHUNK_S" + Twine(inputSource.sourceId) + "_A" +
         Twine(inputSource.sourceAxis) +
         (inputSource.derived ? "_DERIVED" : ""))
            .str();
    auto chunk = getOrCreatePhysicalParameter(
        kernel, chunkName, ParameterRole::Reduction,
        ParameterCategory::Histogram,
        valuesType.getElementType().getIntOrFloatBitWidth(),
        {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
        ParameterBindingAttr::get(kernel.getContext(),
            IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), *inputDimension),
            {}, {}, {}, false, false));
    if (failed(chunk))
      return failure();

    PhysicalExprAttr chunkExtent = expression(
        kernel.getContext(), PhysicalExprKind::Parameter, 0, chunkName);
    auto inputRangeType = cast<FragmentType>(inputRange.getResult().getType());
    auto blockedInputType = FragmentType::get(
        kernel.getContext(), inputRangeType.getElementType(),
        ArrayAttr::get(kernel.getContext(), {chunkExtent}),
        inputRangeType.getAxisMaps(), inputRangeType.getValidity(),
        inputRangeType.getOwner());

    Block *outputBlock = stores.front()->getBlock();
    if (llvm::any_of(stores, [&](StoreOp store) {
          return store->getBlock() != outputBlock;
        }))
      return histogram.emitOpError(
          "histogram output effects do not share one physical control block");
    OpBuilder builder(stores.front());
    auto countType = dyn_cast<IntegerType>(outputType->getElementType());
    if (!countType)
      return histogram.emitOpError(
          "histogram count type has no integer zero identity");
    Value zeroScalar = builder.create<arith::ConstantOp>(
        histogram.getLoc(), countType, builder.getIntegerAttr(countType, 0));
    Value zero = builder.create<SplatOp>(histogram.getLoc(), *outputType,
                                         zeroScalar);
    Value loopStep = builder.create<BinaryOp>(
        histogram.getLoc(), builder.getIndexType(),
        materializeParameter(builder, histogram.getLoc(), *chunk),
        inputRange.getStep(), BinaryOperator::Multiply);
    bool bodyFailed = false;
    std::string failureReason;
    auto loop = builder.create<scf::ForOp>(
        histogram.getLoc(), inputRange.getStart(), inputRange.getLogicalStop(),
        loopStep, ValueRange{zero},
        [](OpBuilder &nested, Location location, Value, ValueRange carries) {
          nested.create<scf::YieldOp>(location, carries);
        });
    // Schema and replay queries inspect the current kernel and loop carries.
    // Attach a complete loop before rebuilding its body: a ForOp construction
    // callback still owns a detached region, without those execution facts.
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    OpBuilder nested(yield);
    auto populate =
        [&](OpBuilder &nested, Location location, Value tileStart,
            ValueRange carries) {
          Value blocked = nested.create<MakeRangeOp>(
              location, blockedInputType, tileStart,
              materializeParameter(nested, location, *chunk),
              inputRange.getStep(), inputRange.getLogicalStart(),
              inputRange.getLogicalStop(), inputRange.getSourceId(),
              inputRange.getSourceAxis(), inputRange.getDerived());
          inheritRangeAuthority(blocked.getDefiningOp(), inputRange);
          Value inputEnd = nested.create<BroadcastOp>(
              location, blockedInputType, inputRange.getLogicalStop());
          auto tailComparison = nested.create<CompareOp>(
              location, predicateType(blockedInputType), blocked, inputEnd,
              ComparePredicate::Lt);
          tailComparison->setAttr(physicalTailAttr, nested.getUnitAttr());
          Value tail = tailComparison.getResult();

          IRMapping mapping;
          mapping.map(inputRange.getResult(), blocked);
          SmallVector<int64_t> traversalDimensions{*inputDimension};
          FailureOr<Value> values = replayPointwiseValue(
              nested, histogram.getValues(), inputSource, traversalDimensions,
              chunkExtent, blocked, tail, histogram.getOperation(), mapping);
          FailureOr<Value> valid = replayPointwiseValue(
              nested, histogram.getValid(), inputSource, traversalDimensions,
              chunkExtent, blocked, tail, histogram.getOperation(), mapping);
          if (failed(values) || failed(valid)) {
            bodyFailed = true;
            failureReason =
                "input values cannot be replayed in the histogram traversal loop";
            return;
          }
          auto blockedValuesType = dyn_cast<FragmentType>((*values).getType());
          if (!blockedValuesType) {
            bodyFailed = true;
            failureReason =
                "replayed histogram values lost their physical fragment schema";
            return;
          }
          FragmentType predicate = predicateType(blockedValuesType);
          FailureOr<Value> projectedValid = projectPhysicalValueToSchema(
              nested, location, *valid, predicate);
          FailureOr<Value> projectedTail =
              projectPhysicalValueToSchema(nested, location, tail, predicate);
          if (failed(projectedValid) || failed(projectedTail)) {
            bodyFailed = true;
            failureReason =
                "histogram input validity cannot adopt its blocked traversal schema";
            return;
          }

          Value outputOffset = nested.create<BinaryOp>(
              location, nested.getIndexType(), outputRange.getStart(),
              outputRange.getLogicalStart(), BinaryOperator::Subtract);
          Value outputEnd = nested.create<BinaryOp>(
              location, nested.getIndexType(), outputOffset,
              outputRange.getExtent(), BinaryOperator::Add);
          Type inputElement = blockedValuesType.getElementType();
          Value typedStart = nested.create<CastOp>(
              location, inputElement, outputOffset);
          Value typedEnd =
              nested.create<CastOp>(location, inputElement, outputEnd);
          FailureOr<Value> start = projectPhysicalValueToSchema(
              nested, location, typedStart, blockedValuesType);
          FailureOr<Value> end = projectPhysicalValueToSchema(
              nested, location, typedEnd, blockedValuesType);
          if (failed(start) || failed(end)) {
            bodyFailed = true;
            failureReason =
                "histogram bin ownership cannot project onto the input values";
            return;
          }
          Value lower = nested.create<CompareOp>(
              location, predicate, *values, *start, ComparePredicate::Ge);
          Value upper = nested.create<CompareOp>(
              location, predicate, *values, *end, ComparePredicate::Lt);
          Value active = nested.create<BinaryOp>(
              location, predicate, *projectedValid, *projectedTail,
              BinaryOperator::LogicalAnd);
          active = nested.create<BinaryOp>(location, predicate, active, lower,
                                           BinaryOperator::LogicalAnd);
          active = nested.create<BinaryOp>(location, predicate, active, upper,
                                           BinaryOperator::LogicalAnd);
          Value localValues = nested.create<BinaryOp>(
              location, blockedValuesType, *values, *start,
              BinaryOperator::Subtract);
          auto partial = nested.create<HistogramOp>(
              location, *outputType, localValues, outputRange.getExtent(),
              active);
          if (Attribute origin = histogram->getAttr(originAttr))
            partial->setAttr(originAttr, origin);
          Value accumulated = nested.create<BinaryOp>(
              location, *outputType, carries.front(), partial.getResult(),
              BinaryOperator::Add);
          yield.getResultsMutable().assign(accumulated);
        };
    populate(nested, histogram.getLoc(), loop.getInductionVar(),
             loop.getRegionIterArgs());
    if (bodyFailed) {
      loop.erase();
      return histogram.emitOpError(
                 "histogram ownership could not be materialized: ")
             << failureReason;
    }
    for (StoreOp store : stores)
      store.getValueMutable().assign(loop.getResult(0));
  }
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult PointwiseRewrite::selectWritebackCandidates(bool accountResourcePressure) {
  IRMapping replayBindings;
  reuseTraversalRanges.clear();
  effectLocalStores.clear();
  llvm::MapVector<Attribute, MakeRangeOp> effectLocalAuthorities;

  SmallVector<StoreOp> candidateStores;
  for (const WriteEffectFacts &effect : writeEffects)
    if (auto store = dyn_cast<StoreOp>(effect.operation)) candidateStores.push_back(store);
  SmallVector<Attribute> postStructuredWritebackKeys;
  reductionCaptureWritebackRanges.clear();
  boundedWritebackRanges.clear();
  for (StoreOp store : candidateStores) {
    for (MakeRangeOp range : allRanges) {
      if (!storeAxisForRange(store, range))
        continue;
      FailureOr<int64_t> dimension = queryRangeDimension(range);
      if (failed(dimension))
        continue;
      PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayAt(
          store.getValue(), sourceAxisIdentity(range),
          PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true,
          store.getOperation(), replayBindings, *dimension);
      bool regionReduction = !replay.structuredPrograms.empty() &&
          llvm::all_of(replay.structuredPrograms, [&](Operation *operation) {
            auto fold = dyn_cast<RegionFoldOp>(operation);
            return fold &&
                   lookupParameter(kernel, fold.getSegment()).getCategory() ==
                       ParameterCategory::RegionReduction;
          });
      if (!replay.isReplayable() || !regionReduction)
        continue;
      FailureOr<Attribute> key = effectLocalKey(range);
      if (succeeded(key) &&
          !llvm::is_contained(postStructuredWritebackKeys, *key))
        postStructuredWritebackKeys.push_back(*key);
    }
  }
  for (StoreOp store : candidateStores) {
    SmallVector<std::pair<MakeRangeOp, int64_t>> candidates;
    llvm::SmallPtrSet<Operation *, 4> capturedRanges;
    int64_t innermostSourceAxis = -1;
    for (MakeRangeOp range : allRanges) {
      // Workset coordinates first need ownership over their complete domain.
      // Their construction-time singleton is not a local writeback traversal.
      if (range->hasAttr(worksetCoordinateRangeAttr) &&
          !isReductionTraversal(range.getOperation()) &&
          !uses.structuredTraversalRanges.contains(range.getOperation()) &&
          !range->hasAttr(sourceSubregionAttr))
        continue;
      // Matrix blocking owns these output occurrences and their operand slices.
      // A matching source in a reduction position is not a writeback dependence.
      if (uses.contractionOwnedStores.contains(store.getOperation()) &&
          isContractionOwned(range.getOperation()))
        continue;
      bool structuredFreeAxis =
          uses.structuredTraversalRanges.contains(range.getOperation()) &&
          !isReductionTraversal(range.getOperation());
      std::optional<int64_t> sourceAxis = storeAxisForRange(store, range);
      if (!sourceAxis)
        continue;
      if (structuredFreeAxis) {
        bool capturesReduction =
            llvm::all_of(writeOperations, [&](Operation *operation) {
              return operation == store.getOperation();
            }) &&
            hasFullRangeReductionCapture(store.getValue(), range,
                                         store.getOperation());
        bool usedByEveryEffect = llvm::all_of(
            writeCoordinates, [&](ArrayRef<Value> coordinates) {
              return llvm::any_of(allRanges, [&](MakeRangeOp candidate) {
                return sameLogicalRange(range, candidate) &&
                       coordinatesUseRange(coordinates, candidate);
              });
            });
        FailureOr<uint64_t> dimension = rangeDimension(range);
        PhysicalReplayFact replay = failed(dimension)
                                        ? PhysicalReplayFact()
                                        : PhysicalProgramAnalysis(kernel)
                                              .replayAt(
                                                  store.getValue(),
                                                  sourceAxisIdentity(range),
                                                  PhysicalReplayScope::ValueGraph,
                                                  /*allowAccesses=*/true,
                                                  store.getOperation(),
                                                  replayBindings,
                                                  static_cast<int64_t>(*dimension));
        PhysicalAxisProjection valueProjection =
            queryFragmentAxis(store.getValue().getType(),
                              sourceAxisIdentity(range));
        bool contractRequiresRange = llvm::any_of(
            replay.contractions, [&](Operation *contract) {
              return contractionFreeAxisNeedsRange(contract, range);
            });
        if ((usedByEveryEffect && !capturesReduction) || failed(dimension) ||
            !valueProjection.isExact() ||
            valueProjection.dimensionId != static_cast<int64_t>(*dimension) ||
            !replay.isReplayable() || replay.crossesStructuredProgram ||
            !contractRequiresRange)
          continue;
        if (capturesReduction) {
          capturedRanges.insert(range.getOperation());
          reductionCaptureWritebackRanges.insert(range.getOperation());
        }
      } else if (hasAccessDependentSubregionBounds(kernel, range)) {
        if (failed(reuseTraversalDimension(kernel, ArrayRef<StoreOp>{store},
                                           range)))
          continue;
      } else {
        std::optional<int64_t> sourceDimension;
        if (FailureOr<uint64_t> dimension = rangeDimension(range);
            succeeded(dimension))
          sourceDimension = *dimension;
        PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayAt(
            store.getValue(), sourceAxisIdentity(range),
            PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true,
            store.getOperation(), replayBindings, sourceDimension);
        FailureOr<Attribute> key = effectLocalKey(range);
        bool postStructuredWriteback =
            succeeded(key) &&
            llvm::is_contained(postStructuredWritebackKeys, *key) &&
            replay.isReplayable();
        bool boundedWriteback = false;
        if (accountResourcePressure && replay.isReplayable() && !replay.crossesStructuredProgram &&
            replay.structuredPrograms.empty() && replay.contractions.empty() &&
            llvm::any_of(writeCoordinates, [&](ArrayRef<Value> coordinates) {
              return !coordinatesUseRange(coordinates, range);
            })) {
          auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
          auto footprint = estimatedFragmentRegisters(kernel, store.getValue());
          // A value filling the entire register file leaves no registers for
          // the write address or other live state. Keep a tunable traversal.
          boundedWriteback = footprint && *footprint >= capabilities.getRegistersPerUnit();
        }
        if (boundedWriteback)
          boundedWritebackRanges.insert(range.getOperation());
        if (!postStructuredWriteback && !boundedWriteback) {
          PhysicalReductionDependencyFact dependency =
              PhysicalProgramAnalysis(kernel).reductionDependency(
                  store.getValue(), sourceAxisIdentity(range), sourceDimension);
          if (!dependency.isExact() || !dependency.depends ||
              dependency.throughStructuredReduction)
            continue;
          reductionCaptureWritebackRanges.insert(range.getOperation());
        }
      }
      candidates.emplace_back(range, *sourceAxis);
      innermostSourceAxis = std::max(innermostSourceAxis, *sourceAxis);
    }
    for (auto [range, sourceAxis] : candidates) {
      deferRange(range.getOperation());
      if (sourceAxis != innermostSourceAxis &&
          !capturedRanges.contains(range.getOperation()))
        continue;
      if (uses.structuredTraversalRanges.contains(range.getOperation()) &&
          !isReductionTraversal(range.getOperation())) {
        FailureOr<Attribute> key = effectLocalKey(range);
        if (failed(key))
          continue;
        MakeRangeOp authority = range;
        auto existing = effectLocalAuthorities.find(*key);
        if (existing != effectLocalAuthorities.end())
          authority = existing->second;
        else
          effectLocalAuthorities[*key] = authority;
        reuseTraversalRanges.insert(authority.getOperation());
        SmallVector<StoreOp> &selected = effectLocalStores[*key];
        if (!llvm::is_contained(selected, store))
          selected.push_back(store);
      } else {
        reuseTraversalRanges.insert(range.getOperation());
      }
    }
  }
  writeTraversalRanges.clear();

  for (Operation *operation : uses.structuredTraversalRanges) {
    auto range = dyn_cast<MakeRangeOp>(operation);
    bool writeOwnership = range &&
        llvm::any_of(writeCoordinates, [&](ArrayRef<Value> coordinates) {
          return llvm::any_of(allRanges, [&](MakeRangeOp candidate) {
            return sharesLogicalTraversal(range, candidate) &&
                   (candidate->hasAttr(sourceSubregionAttr)
                        ? coordinatesDirectlyUseRange(coordinates, candidate)
                        : coordinatesUseRange(coordinates, candidate));
          });
        });
    if (writeOwnership)
      writeTraversalRanges.insert(operation);
    if (isReductionTraversal(operation) || !writeOwnership)
      deferRange(operation);
  }
  {
    llvm::SmallPtrSet<Operation *, 32> seen;
    llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
      return !seen.insert(range.getOperation()).second;
    });
  }
  return success();
}


LogicalResult PointwiseRewrite::realizeWritebacks() {
  IRMapping replayBindings;
  SmallVector<MakeRangeOp> realized;
  SmallVector<MakeRangeOp> retainedFragments;
  for (MakeRangeOp range : dynamicRanges) {
    if (!reuseTraversalRanges.contains(range.getOperation()))
      continue;
    bool useReplayTraversal = true;
    SmallVector<StoreOp> currentStores;
    SmallVector<LoadOp> retainedReads;
    FailureOr<Attribute> effectKey = effectLocalKey(range);
    auto effectLocal = succeeded(effectKey)
                           ? effectLocalStores.find(*effectKey)
                           : effectLocalStores.end();
    if (effectLocal != effectLocalStores.end()) {
      currentStores = effectLocal->second;
    } else {
      kernel.walk([&](StoreOp store) {
        if (storeUsesRange(store, range))
          currentStores.push_back(store);
      });
    }
    if (currentStores.empty())
      continue;
    FailureOr<SmallVector<int64_t>> traversalDimensions =
        traversalDimensionsForStores(currentStores, range);
    if (failed(traversalDimensions)) {
      retainedFragments.push_back(range);
      continue;
    }
    for (StoreOp store : currentStores) {
      PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                                range.getDerived()};
      SmallVector<int64_t> valueDimensions;
      collectTraversalDimensions(store.getValue().getType(), source,
                                 valueDimensions);
      for (int64_t dimension : valueDimensions) {
        if (!llvm::is_contained(*traversalDimensions, dimension))
          continue;
        PhysicalProgramAnalysis analysis(kernel);
        PhysicalReplayFact replay = analysis.replayAt(
            store.getValue(), source, PhysicalReplayScope::ValueGraph,
            /*allowAccesses=*/true, store.getOperation(), replayBindings,
            dimension);
        ReplayPolicy reuse(kernel, ValueRange{store.getValue()}, {store.getOperation()});
        bool materializedFork =
            replay.crossesAccess && reuse.preservesSharedTraversal(
                                        store.getValue(), source, dimension);
        auto reduction = analysis.reductionDependency(store.getValue(), source, dimension);
        // A bounded value that already depends on the complete traversal is
        // retained SSA. Reblocking its writeback needlessly rebuilds that graph.
        bool retainedReduction = replay.isReplayable() &&
            reduction.isExact() && reduction.depends;
        if (retainedReduction) {
          auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
          auto footprint = estimatedFragmentRegisters(kernel, store.getValue());
          retainedReduction = footprint &&
              *footprint < capabilities.getRegistersPerUnit();
        }
        // Replaying a read of the destination also crosses its writes from
        // earlier chunks. Preserve that snapshot unless all addresses agree.
        bool clobbersLaterChunk = llvm::any_of(
            replay.accesses, [&](Operation *access) {
              auto load = dyn_cast<LoadOp>(access);
              if (!load || load.getResource() != store.getResource() ||
                  canReplayReadAt(load, store->getNextNode()))
                return false;
              if (load.getSourceAxes().size() != store.getSourceAxes().size())
                return true;
              auto coordinate =
                  queryCoordinateIndex(store.getCoordinates(), source, dimension);
              if (!coordinate.isExact() || coordinate.dimensionId != dimension)
                return true;
              bool sameCoordinates = llvm::all_of(llvm::enumerate(store.getSourceAxes()),
                  [&](auto axis) {
                    auto readAxis = llvm::find(load.getSourceAxes(), axis.value());
                    if (readAxis == load.getSourceAxes().end())
                      return false;
                    Value lhs = load.getCoordinates()[readAxis - load.getSourceAxes().begin()];
                    Value rhs = store.getCoordinates()[axis.index()];
                    if (samePhysicalScalarExpression(lhs, rhs))
                      return true;
                    auto left = lhs.getDefiningOp<MakeRangeOp>();
                    auto right = rhs.getDefiningOp<MakeRangeOp>();
                    return left && right && sameLogicalRange(left, right) &&
                        samePhysicalScalarExpression(left.getStart(), right.getStart()) &&
                        samePhysicalScalarExpression(left.getExtent(), right.getExtent());
                  });
              if (sameCoordinates)
                return false;
              // Keep a small captured row/column as its original SSA snapshot.
              // Re-reading it after an earlier chunk writes could observe a
              // different value, including through an overlapping view layout.
              auto type = dyn_cast<FragmentType>(load.getType());
              auto selected = type ? queryFragmentAxis(type, source)
                                   : PhysicalAxisProjection{};
              auto footprint = estimatedFragmentRegisters(kernel, load.getResult());
              auto authority = selected.isExact()
                  ? queryExactLogicalRange(analysis.axisRanges(load.getResult(), selected.fragmentAxis))
                  : FailureOr<MakeRangeOp>(failure());
              if (type && type.getShape().size() == 1 && selected.isExact() &&
                  selected.dimensionId == dimension && succeeded(authority) &&
                  isUnitStepRange(*authority) && sameLogicalRange(*authority, range) &&
                  footprint && *footprint < kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr).getRegistersPerUnit()) {
                if (!llvm::is_contained(retainedReads, load))
                  retainedReads.push_back(load);
                return false;
              }
              return true;
            });
        if (clobbersLaterChunk) {
          for (PhysicalAxisProjection projection :
               queryFragmentAxes(store.getValue().getType(), source))
            if (projection.dimensionId == dimension &&
                failed(requireFullDimensionCoverage(
                    kernel, store.getValue(), projection.fragmentAxis)))
              return failure();
          useReplayTraversal = false;
          continue;
        }
        useReplayTraversal &=
            replay.isReplayable() && !materializedFork && !retainedReduction &&
            (effectLocal == effectLocalStores.end() ||
             !replay.crossesStructuredProgram);
      }
    }
    // Keep non-replayable control and shared reduction producers intact.
    // The internal-range path below binds their full-coverage extent,
    // avoiding control cloning and duplicate costly fragment computation.
    if (!useReplayTraversal) {
      retainedFragments.push_back(range);
      continue;
    }
    // Close snapshots before constructing slices with the same axis identity.
    for (LoadOp read : retainedReads)
      if (failed(realizeFullCoverageDimension(kernel, read.getResult(), 0)))
        return failure();
    if (failed(realizeReusePointwiseTraversal(
            kernel, range, currentStores,
            effectLocal != effectLocalStores.end(), retainedReads,
            [&](StoreOp original, StoreOp replacement) {
              for (auto &entry : effectLocalStores)
                for (StoreOp &selected : entry.second)
                  if (selected == original) selected = replacement;
            })))
      return failure();
    realized.push_back(range);
  }
  for (MakeRangeOp range : retainedFragments) {
    reuseTraversalRanges.erase(range.getOperation());
    deferRange(range.getOperation());
  }
  llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
    return llvm::is_contained(realized, range);
  });
  eraseDeadPhysicalValues(kernel);
  refreshProgramFacts();
  return success();
}


} // namespace intent::gpu::pointwise
