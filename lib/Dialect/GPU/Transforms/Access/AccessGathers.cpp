#include "AccessComposition.h"
#include "../Value/ReplayPolicy.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::gpu::access {

FailureOr<bool> composeReshapedGather(GatherOp gather) {
  auto reshape = gather.getSource().getDefiningOp<ReshapeOp>();
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!reshape)
    return false;
  auto source = cast<FragmentType>(reshape.getValue().getType());
  auto shaped = cast<FragmentType>(reshape.getResult().getType());
  auto relations = queryFragmentOperandRelations(reshape);
  if (failed(relations))
    return false;
  const auto &groups = relations->front().groups;
  // Scalar indexing is a rectangular extraction in the reshaped layout.
  // Flattening it can turn a native split into a strided fragment gather.
  if (result && !gather.getCoordinates().empty() &&
      llvm::all_of(llvm::zip(gather.getCoordinates(), gather.getSourceAxes()),
                   [&](auto entry) {
        Value coordinate = std::get<0>(entry);
        int64_t axis = std::get<1>(entry);
        while (isa<FragmentType>(coordinate.getType())) {
          UniformExpression expression = describeUniformValue(coordinate);
          if (expression.kind != UniformKind::Forward ||
              expression.operands.size() != 1)
            break;
          coordinate = expression.operands.front();
        }
        if (!isa<FragmentType>(coordinate.getType()))
          return true;
        auto range = coordinate.getDefiningOp<MakeRangeOp>();
        return range && isZero(range.getStart()) && isUnitStepRange(range) &&
               queryLaunchExpression(range.getExtent()) ==
                   shaped.getShape()[axis];
      }))
    return false;
  if (source.getShape() == shaped.getShape() &&
      llvm::all_of(groups, [](const FragmentAxisGroup &group) {
        return group.sourceAxes.size() == 1 && group.resultAxes.size() == 1 &&
               group.sourceAxes.front() == group.resultAxes.front();
      }))
    return false;
  PhysicalProgramAnalysis analysis(gather->getParentOfType<func::FuncOp>());
  for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
    if (analysis.axisRealization(reshape.getValue(), axis).constructionScalarSeed)
      return false;
  for (unsigned axis = 0; axis < shaped.getShape().size(); ++axis)
    if (analysis.axisRealization(reshape.getResult(), axis).constructionScalarSeed)
      return false;
  for (const FragmentAxisGroup &group : groups) {
    if (group.sourceAxes.size() == 1 && group.resultAxes.size() == 1)
      continue;
    for (auto [type, axes] :
         {std::pair{source, ArrayRef<unsigned>(group.sourceAxes)},
          std::pair{shaped, ArrayRef<unsigned>(group.resultAxes)}})
      for (unsigned axis : axes) {
        auto extent = cast<PhysicalExprAttr>(type.getShape()[axis]);
        if (extent.getKind() != PhysicalExprKind::Constant ||
            extent.getValue() <= 0)
          return false;
      }
  }
  SmallVector<Value> coordinates(shaped.getShape().size());
  for (auto [coordinate, axis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
    coordinates[axis] = coordinate;
  SmallVector<unsigned> retainedAxes(shaped.getShape().size());
  for (unsigned axis = 0; axis < coordinates.size(); ++axis) {
    if (coordinates[axis])
      continue;
    if (!result)
      return false;
    auto mapping = cast<AxisMapAttr>(shaped.getAxisMaps()[axis]);
    auto projection = queryFragmentAxis(result, sourceAxisIdentity(mapping),
                                        mapping.getDimensionId());
    if (!projection.isExact() || projection.dimensionId != mapping.getDimensionId() ||
        result.getShape()[projection.fragmentAxis] != shaped.getShape()[axis])
      return false;
    retainedAxes[axis] = projection.fragmentAxis;
  }
  OpBuilder builder(gather);
  Location location = gather.getLoc();
  Type indexType = builder.getIndexType();
  if (result)
    indexType = FragmentType::get(
        result.getContext(), builder.getIndexType(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
  auto indexValue = [&](Value value) -> Value {
    return result ? Value(builder.create<SplatOp>(location, indexType, value))
                  : value;
  };
  auto extentValue = [&](Attribute attribute) -> Value {
    auto extent = cast<PhysicalExprAttr>(attribute);
    if (extent.getKind() == PhysicalExprKind::Constant)
      return builder.create<arith::ConstantIndexOp>(location, extent.getValue());
    return builder.create<PhysicalExprOp>(location, builder.getIndexType(), extent);
  };
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  for (unsigned axis = 0; axis < coordinates.size(); ++axis) {
    Value coordinate = coordinates[axis];
    if (!coordinate) {
      auto mapping = cast<AxisMapAttr>(result.getAxisMaps()[retainedAxes[axis]]);
      auto type = FragmentType::get(
          result.getContext(), builder.getIndexType(),
          builder.getArrayAttr({shaped.getShape()[axis]}),
          builder.getArrayAttr({AxisMapAttr::get(
              result.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
              mapping.getDimensionId(), 0, mapping.getDerived())}),
          result.getValidity(), result.getOwner());
      Value extent = extentValue(shaped.getShape()[axis]);
      coordinate = builder.create<MakeRangeOp>(
          location, type, zero, extent, one, zero, extent,
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
    }
    Type element = coordinate.getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    if (!element.isIndex()) {
      Type target = builder.getIndexType();
      if (auto fragment = dyn_cast<FragmentType>(coordinate.getType()))
        target = FragmentType::get(builder.getContext(), builder.getIndexType(),
            fragment.getShape(), fragment.getAxisMaps(), fragment.getValidity(),
            fragment.getOwner());
      coordinate = builder.create<CastOp>(location, target, coordinate);
    }
    auto projected = projectPhysicalValueToSchema(builder, location, coordinate, indexType);
    if (failed(projected))
      return gather.emitOpError("reshape gather lost an exact result coordinate projection");
    coordinates[axis] = *projected;
  }
  SmallVector<Value> selected(source.getShape().size());
  for (const FragmentAxisGroup &group : groups) {
    if (group.sourceAxes.empty())
      continue;
    if (group.sourceAxes.size() == 1 && group.resultAxes.size() == 1) {
      selected[group.sourceAxes.front()] = coordinates[group.resultAxes.front()];
      continue;
    }
    Value ordinal = indexValue(zero);
    for (unsigned axis : group.resultAxes) {
      Value extent = indexValue(extentValue(shaped.getShape()[axis]));
      ordinal = builder.create<BinaryOp>(location, indexType, ordinal, extent,
                                        BinaryOperator::Multiply);
      ordinal = builder.create<BinaryOp>(location, indexType, ordinal,
          coordinates[axis], BinaryOperator::Add);
    }
    ArrayRef<unsigned> axes = group.sourceAxes;
    for (unsigned position = axes.size(); position-- > 0;) {
      unsigned axis = axes[position];
      Value coordinate = ordinal;
      if (position != 0) {
        Value extent = indexValue(extentValue(source.getShape()[axis]));
        coordinate = builder.create<BinaryOp>(location, indexType, ordinal, extent,
                                              BinaryOperator::Remainder);
        ordinal = builder.create<BinaryOp>(location, indexType, ordinal, extent,
                                           BinaryOperator::FloorDivide);
      }
      selected[axis] = coordinate;
    }
  }
  SmallVector<int64_t> axes;
  Type predicate = builder.getI1Type();
  if (result)
    predicate = FragmentType::get(result.getContext(), builder.getI1Type(),
        result.getShape(), result.getAxisMaps(), result.getValidity(), result.getOwner());
  Value valid = gather.getValid();
  Value lower = indexValue(zero);
  for (unsigned axis = 0; axis < selected.size(); ++axis)
  {
    axes.push_back(axis);
    Value upper = indexValue(extentValue(source.getShape()[axis]));
    Value nonnegative = builder.create<CompareOp>(location, predicate,
        selected[axis], lower, ComparePredicate::Ge);
    Value below = builder.create<CompareOp>(location, predicate,
        selected[axis], upper, ComparePredicate::Lt);
    Value bounded = builder.create<BinaryOp>(location, predicate, nonnegative,
                                            below, BinaryOperator::LogicalAnd);
    auto combined = combinePredicates(builder, location, gather.getResult().getType(), valid, bounded);
    if (failed(combined))
      return failure();
    valid = *combined;
  }
  Value fill = gather.getFill();
  if (!fill) {
    if (result) {
      auto zeroFill = materializeZeroFragment(builder, location, result);
      if (failed(zeroFill))
        return failure();
      fill = *zeroFill;
    } else {
      fill = builder.create<arith::ConstantOp>(
          location, builder.getZeroAttr(gather.getResult().getType()));
    }
  }
  auto replacement = builder.create<GatherOp>(location, gather.getResult().getType(), reshape.getValue(),
      selected, valid, fill, axes);
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  return true;
}

FailureOr<bool> composeRangeGather(GatherOp gather) {
  auto range = gather.getSource().getDefiningOp<MakeRangeOp>();
  Type result = gather.getResult().getType();
  auto fragment = dyn_cast<FragmentType>(result);
  if (!range || (!fragment && !result.isIndex()) ||
      !uniformElementType(result).isIndex() ||
      gather.getSourceAxes() != ArrayRef<int64_t>{0} ||
      gather.getCoordinates().size() != 1 ||
      !uniformElementType(gather.getCoordinates().front().getType()).isIndex())
    return false;
  OpBuilder builder(gather);
  Location location = gather.getLoc();
  auto coordinate = projectPhysicalValueToSchema(
      builder, location, gather.getCoordinates().front(), result);
  if (failed(coordinate))
    return gather.emitOpError("range gather lost its result coordinate projection");
  auto projectScalar = [&](Value value) -> Value {
    return fragment ? builder.create<SplatOp>(location, fragment, value).getResult()
                    : value;
  };
  Value start = projectScalar(range.getStart());
  Value step = projectScalar(range.getStep());
  Value offset = builder.create<BinaryOp>(location, result, *coordinate, step,
                                         BinaryOperator::Multiply);
  Value value = builder.create<BinaryOp>(location, result, start, offset,
                                        BinaryOperator::Add);
  if (gather.getValid())
    value = builder.create<SelectOp>(location, result, gather.getValid(), value,
                                     gather.getFill());
  gather.getResult().replaceAllUsesWith(value);
  gather.erase();
  return true;
}

FailureOr<bool> composePointwiseGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  Operation *producer = gather.getSource().getDefiningOp();
  if (!source || !producer ||
      !isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
           BroadcastOp, SplatOp, ReshapeOp>(producer) ||
      gather.getSourceAxes().size() != source.getShape().size())
    return false;
  ReplayPolicy reuse(gather->getParentOfType<func::FuncOp>(),
                     ValueRange{gather.getSource()}, {gather.getOperation()},
                     [&](Value value) { return value == gather.getSource(); });
  IRMapping bindings;
  for (Value operand : producer->getOperands()) bindings.map(operand, operand);
  if (reuse.duplicatesExpensiveWork(gather.getSource(), &bindings)) return false;
  if (auto binary = dyn_cast<BinaryOp>(producer);
      binary && !isa<FloatType>(source.getElementType())) {
    // Masked operands below are filled with zero. Do not introduce undefined
    // integer arithmetic on those inactive lanes.
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
    case BinaryOperator::Subtract:
    case BinaryOperator::Multiply:
    case BinaryOperator::Maximum:
    case BinaryOperator::Minimum:
    case BinaryOperator::LogicalAnd:
    case BinaryOperator::LogicalOr:
    case BinaryOperator::BitwiseAnd:
    case BinaryOperator::BitwiseOr:
    case BinaryOperator::BitwiseXor:
      break;
    default:
      return false;
    }
  }
  bool positionalRebind = false;
  if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
    auto input = cast<FragmentType>(reshape.getValue().getType());
    positionalRebind = input.getShape() == source.getShape() &&
        llvm::all_of(reshape.getReassociation(), [](Attribute attribute) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          return group.getSourceAxes().size() == 1 &&
                 group.getResultAxes().size() == 1 &&
                 group.getSourceAxes()[0] == group.getResultAxes()[0];
        });
    if (!positionalRebind)
      return false;
  }
  SmallVector<Value> coordinates(source.getShape().size());
  for (auto [coordinate, axis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (axis < 0 || axis >= static_cast<int64_t>(coordinates.size()) ||
        coordinates[axis])
      return false;
    coordinates[axis] = coordinate;
  }
  if (llvm::any_of(coordinates, [](Value value) { return !value; }))
    return false;
  PhysicalProgramAnalysis analysis(gather->getParentOfType<func::FuncOp>());
  auto broadcastUnit = [&](Value operand, unsigned axis) {
    auto input = cast<FragmentType>(operand.getType());
    auto extent = cast<PhysicalExprAttr>(input.getShape()[axis]);
    if (extent.getKind() != PhysicalExprKind::Constant ||
        extent.getValue() != 1)
      return false;
    PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
    return ranges.isExact() && llvm::all_of(ranges.roots, [](MakeRangeOp range) {
      return isProvablySingletonLogicalRange(range);
    });
  };
  SmallVector<SmallVector<unsigned>> operandAxes;
  for (Value operand : producer->getOperands()) {
    SmallVector<unsigned> axes;
    if (auto input = dyn_cast<FragmentType>(operand.getType())) {
      if (positionalRebind) {
        for (unsigned axis = 0; axis < input.getShape().size(); ++axis)
          axes.push_back(axis);
        operandAxes.push_back(std::move(axes));
        continue;
      }
      BroadcastProjection projection = queryAxisProjection(input, source);
      if (!projection.isExact())
        return false;
      for (unsigned inputAxis = 0; inputAxis < input.getShape().size(); ++inputAxis) {
        std::optional<unsigned> selected;
        for (auto [axis, mapped] : llvm::enumerate(projection.targetToSource))
          if (mapped && *mapped == inputAxis) {
            if (selected)
              return false;
            selected = axis;
          }
        if (!selected)
          return false;
        if (!broadcastUnit(operand, inputAxis) &&
            input.getShape()[inputAxis] != source.getShape()[*selected])
          return false;
        axes.push_back(*selected);
      }
    }
    operandAxes.push_back(std::move(axes));
  }
  OpBuilder builder(gather);
  IRMapping mapping;
  auto selectedTypeFor = [&](Type element) -> Type {
    return result ? Type(FragmentType::get(
                        result.getContext(), element, result.getShape(),
                        result.getAxisMaps(), result.getValidity(), result.getOwner()))
                  : element;
  };
  for (auto [operand, axes] : llvm::zip(producer->getOperands(), operandAxes)) {
    auto input = dyn_cast<FragmentType>(operand.getType());
    if (!input || mapping.contains(operand))
      continue;
    Type selectedType = selectedTypeFor(input.getElementType());
    SmallVector<Value> selectedCoordinates;
    SmallVector<int64_t> selectedAxes;
    for (auto [axis, sourceAxis] : llvm::enumerate(axes)) {
      Value coordinate = coordinates[sourceAxis];
      if (broadcastUnit(operand, axis))
        coordinate = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
      Type element = coordinate.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      Type indexType = selectedTypeFor(element);
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, gather.getLoc(), coordinate, indexType);
      if (failed(projected))
        return gather.emitOpError("pointwise gather index lost its result projection");
      selectedCoordinates.push_back(*projected);
      selectedAxes.push_back(axis);
    }
    Value fill;
    if (gather.getValid()) {
      if (auto fragment = dyn_cast<FragmentType>(selectedType)) {
        FailureOr<Value> zero = materializeZeroFragment(builder, gather.getLoc(), fragment);
        if (failed(zero))
          return failure();
        fill = *zero;
      } else {
        fill = builder.create<arith::ConstantOp>(gather.getLoc(),
                                                builder.getZeroAttr(selectedType));
      }
    }
    mapping.map(operand, builder.create<GatherOp>(
        gather.getLoc(), selectedType, operand, selectedCoordinates,
        gather.getValid(), fill, selectedAxes).getResult());
  }
  // Keep reductions and immutable reads as captured SSA producers. Only the
  // pure pointwise suffix moves to the selected coordinates. Inactive operands
  // use zero (also a valid cast input); restore the original gather fill after
  // evaluating the unchanged typed operation.
  Value value;
  Type resultType = gather.getResult().getType();
  if (positionalRebind || isa<BroadcastOp, SplatOp>(producer)) {
    auto projected = projectPhysicalValueToSchema(
        builder, gather.getLoc(), mapping.lookupOrDefault(producer->getOperand(0)),
        resultType);
    if (failed(projected))
      return failure();
    value = *projected;
  } else {
    auto replacement = cloneWithSchema(builder, producer, mapping, TypeRange{resultType});
    if (failed(replacement))
      return failure();
    value = replacement->front();
  }
  if (gather.getValid())
    value = builder.create<SelectOp>(gather.getLoc(), resultType, gather.getValid(),
                                     value, gather.getFill());
  gather.getResult().replaceAllUsesWith(value);
  gather.erase();
  return true;
}

} // namespace intent::gpu::access
