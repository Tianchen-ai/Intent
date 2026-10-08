#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Value/SchemaMutation.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
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

namespace {

FailureOr<Value> projectPredicate(OpBuilder &builder, Location location,
                                  Value predicate, FragmentType target,
                                  unsigned axis) {
  auto base = dyn_cast<FragmentType>(predicate.getType());
  if (!base || base.getShape().size() != 1 || axis >= target.getShape().size())
    return failure();
  SmallVector<Attribute> shape(
      target.getShape().size(),
      PhysicalExprAttr::get(
          target.getContext(),
          PhysicalExprKind::Constant, 1,
          StringAttr::get(target.getContext()),
          ArrayAttr::get(target.getContext(), {})));
  shape[axis] = base.getShape()[0];
  auto reshaped = FragmentType::get(
      target.getContext(), builder.getI1Type(),
      ArrayAttr::get(target.getContext(), shape), target.getAxisMaps(),
      target.getValidity(), target.getOwner());
  Value result = predicate;
  if (base != reshaped) {
    SmallVector<Attribute> groups;
    for (unsigned resultAxis = 0; resultAxis < shape.size(); ++resultAxis) {
      SmallVector<int64_t> sourceAxes;
      if (resultAxis == axis)
        sourceAxes.push_back(0);
      groups.push_back(ReshapeGroupAttr::get(
          target.getContext(), builder.getDenseI64ArrayAttr(sourceAxes),
          builder.getDenseI64ArrayAttr({static_cast<int64_t>(resultAxis)})));
    }
    result = builder.create<ReshapeOp>(location, reshaped, result,
                                      builder.getArrayAttr(groups));
  }
  auto projected = FragmentType::get(
      target.getContext(), builder.getI1Type(), target.getShape(),
      target.getAxisMaps(), target.getValidity(), target.getOwner());
  if (reshaped != projected)
    result = builder.create<BroadcastOp>(location, projected, result);
  return result;
}

FailureOr<Value> zeroValue(OpBuilder &builder, Location location, Type type) {
  auto fragment = dyn_cast<FragmentType>(type);
  Type element = fragment ? fragment.getElementType() : type;
  TypedAttr zero;
  if (auto integer = dyn_cast<IntegerType>(element))
    zero = builder.getIntegerAttr(integer, 0);
  else if (auto floating = dyn_cast<FloatType>(element))
    zero = builder.getFloatAttr(floating, 0.0);
  else if (isa<IndexType>(element))
    zero = builder.getIndexAttr(0);
  if (!zero)
    return failure();
  FailureOr<Value> scalar =
      materializeScalarConstant(builder, location, zero, element);
  if (failed(scalar))
    return failure();
  if (!fragment)
    return *scalar;
  return Value(builder.create<BroadcastOp>(location, fragment, *scalar));
}


} // namespace

FailureOr<Value> materializeAccessCoordinate(OpBuilder &builder,
                                             AccessOpInterface access,
                                             unsigned coordinateIndex) {
  auto projection = queryAccessCoordinateProjection(access, coordinateIndex);
  if (!projection.isExact())
    return access.emitOpError("coordinate has no exact access operand projection");
  Value value = access.getAccessCoordinates()[coordinateIndex];
  auto source = dyn_cast<FragmentType>(value.getType());
  if (!source)
    return value;
  auto domain = cast<FragmentType>(access.getAccessValueType());
  auto target = FragmentType::get(
      domain.getContext(), source.getElementType(), domain.getShape(),
      domain.getAxisMaps(), domain.getValidity(), domain.getOwner());
  if (source == target)
    return value;

  Attribute origin;
  if (Operation *producer = value.getDefiningOp())
    origin = producer->getAttr(originAttr);
  auto rememberOrigin = [&](Operation *operation) {
    if (origin)
      operation->setAttr(originAttr, origin);
  };
  SmallVector<int64_t> order;
  SmallVector<Attribute> shape, groups;
  auto one = PhysicalExprAttr::get(
      domain.getContext(), PhysicalExprKind::Constant, 1,
      builder.getStringAttr(""), builder.getArrayAttr({}));
  for (auto [axis, sourceAxis] : llvm::enumerate(projection.targetToSource)) {
    SmallVector<int64_t> sources;
    if (sourceAxis) {
      sources.push_back(order.size());
      order.push_back(*sourceAxis);
    }
    shape.push_back(sourceAxis ? source.getShape()[*sourceAxis] : Attribute(one));
    groups.push_back(ReshapeGroupAttr::get(
        domain.getContext(), builder.getDenseI64ArrayAttr(sources),
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(axis)})));
  }
  if (!llvm::all_of(llvm::enumerate(order), [](auto item) {
        return item.index() == static_cast<unsigned>(item.value());
      })) {
    SmallVector<Attribute> permutedShape, mappings;
    for (auto [axis, from] : llvm::enumerate(order)) {
      permutedShape.push_back(source.getShape()[from]);
      auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[from]);
      mappings.push_back(AxisMapAttr::get(
          domain.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), axis, mapping.getDerived()));
    }
    auto type = FragmentType::get(
        domain.getContext(), source.getElementType(), builder.getArrayAttr(permutedShape),
        builder.getArrayAttr(mappings), source.getValidity(), source.getOwner());
    auto transpose = builder.create<TransposeOp>(access.getLoc(), type, value, order);
    rememberOrigin(transpose);
    value = transpose.getResult();
  }
  auto expanded = FragmentType::get(
      domain.getContext(), source.getElementType(), builder.getArrayAttr(shape),
      domain.getAxisMaps(), domain.getValidity(), domain.getOwner());
  if (value.getType() != expanded) {
    auto reshape = builder.create<ReshapeOp>(
        access.getLoc(), expanded, value, builder.getArrayAttr(groups));
    rememberOrigin(reshape);
    value = reshape.getResult();
  }
  if (expanded != target) {
    auto broadcast = builder.create<BroadcastOp>(access.getLoc(), target, value);
    rememberOrigin(broadcast);
    value = broadcast.getResult();
  }
  return value;
}

FailureOr<Value> materializeNonOverlappingView(func::FuncOp kernel,
                                             Value resource) {
  auto view = cast<ViewType>(resource.getType());
  SmallVector<OpFoldResult> strides;
  for (Attribute attribute : view.getLayout().getStrides()) {
    auto expression = cast<PhysicalExprAttr>(attribute);
    if (expression.getKind() == PhysicalExprKind::Constant) {
      strides.push_back(IntegerAttr::get(IndexType::get(kernel.getContext()), expression.getValue()));
      continue;
    }
    Value stride = resolveArgument(kernel, expression.getArgumentReference());
    if (!stride || !stride.getType().isIndex())
      return failure();
    strides.push_back(stride);
  }
  if (strides.empty())
    return failure();
  OpBuilder builder(&kernel.front(), kernel.front().begin());
  Location location = resource.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  SmallVector<Value> values;
  for (OpFoldResult stride : strides)
    values.push_back(isa<Value>(stride) ? cast<Value>(stride) : Value(
        builder.create<arith::ConstantIndexOp>(
            location, cast<IntegerAttr>(cast<Attribute>(stride)).getInt())));
  Value valid = builder.create<CompareOp>(location, builder.getI1Type(),
                                         values.back(), zero, ComparePredicate::Gt);
  for (unsigned axis = values.size() - 1; axis > 0; --axis) {
    Value divisor = builder.create<BinaryOp>(location, builder.getIndexType(),
        values[axis], one, BinaryOperator::Maximum);
    Value slots = builder.create<BinaryOp>(location, builder.getIndexType(),
        values[axis - 1], divisor, BinaryOperator::FloorDivide);
    Value extent = builder.create<DimOp>(location, builder.getIndexType(),
                                         resource, axis);
    extent = builder.create<BinaryOp>(location, builder.getIndexType(), extent,
                                       one, BinaryOperator::Maximum);
    // Division avoids overflow in stride >= extent * next_stride. Every
    // accepted suffix has positive strides and fits inside its outer stride.
    Value separate = builder.create<CompareOp>(location, builder.getI1Type(),
                                               slots, extent, ComparePredicate::Ge);
    valid = builder.create<BinaryOp>(location, builder.getI1Type(), valid,
                                      separate, BinaryOperator::LogicalAnd);
  }
  return valid;
}

FailureOr<Value> materializeScalarConstant(OpBuilder &builder,
                                           Location location, Attribute value,
                                           Type resultType) {
  if (isa<IndexType>(resultType)) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer)
      return failure();
    return Value(
        builder.create<arith::ConstantIndexOp>(location, integer.getInt()));
  }
  if (auto integerType = dyn_cast<IntegerType>(resultType)) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer)
      return failure();
    auto signless = IntegerType::get(builder.getContext(), integerType.getWidth());
    llvm::APInt bits = integer.getValue().sextOrTrunc(integerType.getWidth());
    Value raw = builder.create<arith::ConstantOp>(
        location, signless, IntegerAttr::get(signless, bits));
    if (integerType.isSignless())
      return raw;
    return Value(builder.create<CastOp>(location, integerType, raw));
  }
  if (auto floatType = dyn_cast<FloatType>(resultType)) {
    auto floating = dyn_cast<FloatAttr>(value);
    if (!floating)
      return failure();
    return Value(builder.create<arith::ConstantOp>(
        location, floatType,
        FloatAttr::get(floatType, floating.getValueAsDouble())));
  }
  return failure();
}

static FailureOr<Value> projectFragmentValue(OpBuilder &builder,
                                             Location location, Value value,
                                             FragmentType target, ValueTypeChangeCallback changed) {
  if (value.getType() == target)
    return value;
  Type element = value.getType();
  auto source = dyn_cast<FragmentType>(element);
  if (source)
    element = source.getElementType();
  if (!isa<IntegerType, FloatType, IndexType>(element) ||
      element != target.getElementType())
    return failure();
  if (auto extract = value.getDefiningOp<ExtractOp>())
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>())
      return projectFragmentValue(builder, location,
                                  record.getFields()[extract.getField()], target, changed);
  Value scalar = value;
  while (isa<FragmentType>(scalar.getType())) {
    UniformExpression expression = describeUniformValue(scalar);
    if (expression.kind != UniformKind::Forward || expression.operands.size() != 1)
      break;
    scalar = expression.operands.front();
  }
  if (scalar.getType() == element)
    return Value(builder.create<SplatOp>(location, target, scalar));
  if (source)
    if (auto constant = dyn_cast_or_null<TypedAttr>(
            UniformValueAnalysis(describeUniformValue).evaluate(value));
        constant && constant.getType() == element) {
      Value scalar = builder.create<arith::ConstantOp>(location, element, constant);
      return Value(builder.create<SplatOp>(location, target, scalar));
    }
  if (source && source.getAxisMaps() != target.getAxisMaps()) {
    auto permutation = queryAxisPermutation(source, target);
    // An exact coordinate permutation is a transpose, not a broadcast. Match
    // source occurrences so equal-sized axes stay distinct.
    if (permutation &&
        llvm::any_of(llvm::enumerate(*permutation), [](auto item) {
          return item.value() != static_cast<int64_t>(item.index());
        })) {
      SmallVector<Attribute> shape;
      for (int64_t axis : *permutation)
        shape.push_back(source.getShape()[axis]);
      auto reordered = FragmentType::get(
          target.getContext(), element, builder.getArrayAttr(shape),
          target.getAxisMaps(), source.getValidity(), source.getOwner());
      auto transpose = builder.create<TransposeOp>(location, reordered, value,
                                                  *permutation);
      if (Operation *definition = value.getDefiningOp())
        if (Attribute origin = definition->getAttr(originAttr))
          transpose->setAttr(originAttr, origin);
      return projectFragmentValue(builder, location, transpose.getResult(),
                                  target, changed);
    }
  }
  Operation *projection = nullptr;
  if (!source) {
    projection = builder.create<SplatOp>(location, target, value);
  } else if (auto splat = value.getDefiningOp<SplatOp>()) {
    projection = builder.create<SplatOp>(location, target, splat.getValue());
  } else if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
    auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
    if (input && input.getShape().size() < target.getShape().size()) {
      auto inputRelations = queryFragmentOperandRelations(broadcast.getOperation());
      auto resultRelation = queryAxisProjection(source, target);
      if (succeeded(inputRelations) && inputRelations->size() == 1 &&
          resultRelation.isExact()) {
        SmallVector<Attribute> shape(input.getShape().getValue());
        for (auto [targetAxis, sourceAxis] :
             llvm::enumerate(resultRelation.targetToSource)) {
          if (!sourceAxis)
            continue;
          const auto *group = inputRelations->front().groupForResultAxis(*sourceAxis);
          if (!group || group->sourceAxes.size() != 1)
            continue;
          unsigned inputAxis = group->sourceAxes.front();
          auto extent = cast<PhysicalExprAttr>(shape[inputAxis]);
          if (extent.getKind() !=
                  PhysicalExprKind::Constant ||
              extent.getValue() != 1)
            shape[inputAxis] = target.getShape()[targetAxis];
        }
        // Retile the producer in its own rank before expanding it. In
        // particular, a reduced row predicate must remain a row reduction.
        auto inputTarget = FragmentType::get(
            target.getContext(), input.getElementType(),
            builder.getArrayAttr(shape), input.getAxisMaps(),
            input.getValidity(), input.getOwner());
        FailureOr<Value> projected = projectFragmentValue(
            builder, location, broadcast.getValue(), inputTarget, changed);
        if (succeeded(projected) &&
            queryBroadcastProjection(inputTarget, target).isExact())
          projection = builder.create<BroadcastOp>(location, target, *projected);
      }
    } else {
      FailureOr<Value> projected =
          projectFragmentValue(builder, location, broadcast.getValue(), target, changed);
      if (succeeded(projected))
        return *projected;
    }
  } else if (auto transpose = value.getDefiningOp<TransposeOp>()) {
    auto input = transpose.getValue().getType();
    auto relation = queryAxisProjection(source, target);
    bool projects = source.getShape().size() == target.getShape().size() &&
        relation.isExact() && llvm::all_of(
            llvm::enumerate(relation.targetToSource), [](auto item) {
              return item.value() && *item.value() == item.index();
            });
    auto relations = queryFragmentOperandRelations(transpose.getOperation());
    auto declaredInput = FragmentType::get(
        target.getContext(), input.getElementType(), input.getShape(),
        input.getAxisMaps(), target.getValidity(), target.getOwner());
    // Transport the selected extents through the original permutation. The
    // producer keeps its own axis occurrences while the result is retiled.
    auto transported = projects && succeeded(relations) && relations->size() == 1
        ? transportFragmentOperandType(relations->front(), target, declaredInput)
        : FailureOr<Type>(failure());
    if (succeeded(transported)) {
      auto projected = projectFragmentValue(
          builder, location, transpose.getValue(), cast<FragmentType>(*transported),
          changed);
      if (succeeded(projected))
        projection = builder.create<TransposeOp>(
            location, target, *projected, transpose.getPermutation());
    }
  } else if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
    auto input = cast<FragmentType>(reshape.getValue().getType());
    bool projects = source.getShape().size() == target.getShape().size();
    BroadcastProjection relation = queryAxisProjection(source, target);
    projects &= relation.isExact() && llvm::all_of(
        llvm::enumerate(relation.targetToSource), [](auto item) {
          return item.value() && *item.value() == item.index();
        });
    auto relations = queryFragmentOperandRelations(reshape.getOperation());
    auto declaredInput = FragmentType::get(
        target.getContext(), input.getElementType(), input.getShape(),
        input.getAxisMaps(), target.getValidity(), target.getOwner());
    auto transported = projects && succeeded(relations) && relations->size() == 1
        ? transportFragmentOperandType(relations->front(), target, declaredInput)
        : FailureOr<Type>(failure());
    if (succeeded(transported)) {
      auto inputTarget = cast<FragmentType>(*transported);
      FailureOr<Value> projected =
          projectFragmentValue(builder, location, reshape.getValue(), inputTarget, changed);
      if (succeeded(projected))
        projection = builder.create<ReshapeOp>(
            location, target, *projected, reshape.getReassociation());
    }
  } else if (auto reduce = value.getDefiningOp<ReduceOp>();
             reduce && reduce.getSources().size() == 1 &&
             reduce.getIdentities().size() == 1 && reduce.getCaptures().size() == 0 &&
             reduce.getNumResults() == 1 &&
             source.getOwner() == target.getOwner() &&
             succeeded(proveLaneWiseHelper(reduce.getCombine()))) {
    auto projected = projectLaneWiseReduction(builder, location, reduce, target);
    if (succeeded(projected)) projection = projected->getDefiningOp();
  }
  if (!projection && queryBroadcastProjection(source, target).isExact())
    projection = builder.create<BroadcastOp>(location, target, value);
  Operation *definition = value.getDefiningOp();
  if (!projection && isLaneWisePointwiseOperation(definition)) {
    IRMapping mapping;
    for (Value operand : definition->getOperands()) {
      Type element = operand.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      auto operandTarget = FragmentType::get(
          target.getContext(), element, target.getShape(), target.getAxisMaps(),
          target.getValidity(), target.getOwner());
      FailureOr<Value> projected =
          projectFragmentValue(builder, location, operand, operandTarget, changed);
      if (failed(projected))
        return failure();
      mapping.map(operand, *projected);
    }
    auto rebuilt = cloneWithSchema(builder, definition, mapping,
                                    TypeRange{target});
    if (failed(rebuilt))
      return failure();
    return rebuilt->front();
  }
  if (!projection)
    return failure();
  if (Operation *definition = value.getDefiningOp())
    if (Attribute origin = definition->getAttr(originAttr))
      projection->setAttr(originAttr, origin);
  return projection->getResult(0);
}

FailureOr<Value> projectPhysicalValueToSchema(OpBuilder &builder,
                                              Location location, Value value,
                                              Type target, ValueTypeChangeCallback changed) {
  if (value.getType() == target)
    return value;
  if (auto fragment = dyn_cast<FragmentType>(target))
    return projectFragmentValue(builder, location, value, fragment, changed);
  auto targetRecord = dyn_cast<RecordType>(target);
  auto sourceRecord = dyn_cast<RecordType>(value.getType());
  if (!targetRecord || !sourceRecord ||
      sourceRecord.getFieldNames() != targetRecord.getFieldNames() ||
      sourceRecord.getFieldTypes().size() !=
          targetRecord.getFieldTypes().size())
    return failure();
  auto record = value.getDefiningOp<MakeRecordOp>();
  SmallVector<Value> projectedFields;
  for (auto [index, targetField] :
       llvm::enumerate(targetRecord.getFieldTypes())) {
    Type sourceType =
        cast<TypeAttr>(sourceRecord.getFieldTypes()[index]).getValue();
    Value field = record
                      ? record.getFields()[index]
                      : Value(builder.create<ExtractOp>(location, sourceType,
                                                        value, index));
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, location, field, cast<TypeAttr>(targetField).getValue(), changed);
    if (failed(projected))
      return failure();
    projectedFields.push_back(*projected);
  }
  auto projected =
      builder.create<MakeRecordOp>(location, targetRecord, projectedFields);
  if (Operation *definition = value.getDefiningOp())
    if (Attribute origin = definition->getAttr(originAttr))
      projected->setAttr(originAttr, origin);
  return projected.getResult();
}

FailureOr<Value> materializeZeroValue(OpBuilder &builder, Location location,
                                     Type target) {
  return zeroValue(builder, location, target);
}

FailureOr<Value> materializeZeroFragment(OpBuilder &builder,
                                         Location location,
                                         FragmentType target) {
  return materializeZeroValue(builder, location, target);
}

FailureOr<Value> projectPredicateToFragmentAxis(OpBuilder &builder,
                                                Location location,
                                                Value predicate,
                                                FragmentType target,
                                                unsigned fragmentAxis) {
  return projectPredicate(builder, location, predicate, target, fragmentAxis);
}

FailureOr<Value> projectPredicateToFragment(OpBuilder &builder,
                                            Location location, Value predicate,
                                            FragmentType target,
                                            PhysicalSourceAxis source) {
  PhysicalAxisProjection projection = queryFragmentAxis(target, source);
  if (!projection.isExact())
    return failure();
  return projectPredicate(builder, location, predicate, target,
                          projection.fragmentAxis);
}

FailureOr<Value> projectPredicateToFragment(OpBuilder &builder,
                                            Location location, Value predicate,
                                            FragmentType target,
                                            int64_t dimensionId) {
  PhysicalDimensionProjection projection =
      queryFragmentDimension(target, dimensionId);
  if (!projection.isExact())
    return failure();
  return projectPredicate(builder, location, predicate, target,
                          projection.fragmentAxis);
}

FailureOr<Value> materializeValidityConjunction(
    OpBuilder &builder, Location location, Value lhs, Value rhs,
    FragmentType valueType) {
  auto predicateType = FragmentType::get(
      valueType.getContext(), builder.getI1Type(), valueType.getShape(),
      valueType.getAxisMaps(), valueType.getValidity(), valueType.getOwner());
  Value result;
  if (lhs) {
    FailureOr<Value> projected =
        projectPhysicalValueToSchema(builder, location, lhs, predicateType);
    if (failed(projected))
      return failure();
    result = *projected;
  }
  if (rhs) {
    FailureOr<Value> projected =
        projectPhysicalValueToSchema(builder, location, rhs, predicateType);
    if (failed(projected))
      return failure();
    result = result ? Value(builder.create<BinaryOp>(
                          location, predicateType, result, *projected,
                          BinaryOperator::LogicalAnd))
                    : *projected;
  }
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

FailureOr<Value> materializeRetargetedValidity(
    OpBuilder &builder, Location location, Value original,
    ArrayRef<std::pair<MakeRangeOp, Value>> originalTailRanges,
    Value physicalTail, FragmentType target) {
  func::FuncOp kernel;
  if (!originalTailRanges.empty())
    kernel = originalTailRanges.front().first->getParentOfType<func::FuncOp>();
  else if (original)
    if (Operation *definition = original.getDefiningOp())
      kernel = definition->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<std::pair<MakeRangeOp, Value>> equivalentTailRanges(
      originalTailRanges.begin(), originalTailRanges.end());
  if (original && isa<FragmentType>(original.getType())) {
    for (auto [expected, end] : originalTailRanges) {
      auto axis =
          queryFragmentAxis(original.getType(), sourceAxisIdentity(expected));
      auto dimension = queryRangeDimension(expected);
      if (!axis.isExact() || failed(dimension) || axis.dimensionId != *dimension)
        continue;
      auto ranges = analysis.axisRanges(original, axis.fragmentAxis);
      if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty())
        continue;
      SmallVector<MakeRangeOp> coincident{expected};
      llvm::append_range(coincident, ranges.roots);
      if (!analysis.lockstepRanges(coincident).isExact())
        continue;
      // A predicate may carry the output's coordinate identity on this input
      // axis. Keep the positional proof as well as equal physical traversal;
      // equal range bounds alone do not equate independent Cartesian axes.
      for (MakeRangeOp range : ranges.roots)
        if (!llvm::is_contained(equivalentTailRanges,
                                std::pair<MakeRangeOp, Value>{range, end}))
          equivalentTailRanges.emplace_back(range, end);
    }
  }

  std::function<FailureOr<Value>(Value)> residual =
      [&](Value value) -> FailureOr<Value> {
    if (!value || analysis.isTailPredicate(value, equivalentTailRanges))
      return Value();
    if (value.getType().isInteger(1))
      return value;
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
      return residual(broadcast.getValue());
    if (auto splat = value.getDefiningOp<SplatOp>())
      return residual(splat.getValue());
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      return residual(reshape.getValue());
    if (auto transpose = value.getDefiningOp<TransposeOp>())
      return residual(transpose.getValue());
    auto conjunction = value.getDefiningOp<BinaryOp>();
    if (!conjunction)
      return failure();
    Type element = conjunction.getResult().getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    bool logical =
        conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
    bool bitwiseI1 =
        conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
        element.isInteger(1);
    if (!logical && !bitwiseI1)
      return failure();
    FailureOr<Value> lhs = residual(conjunction.getLhs());
    FailureOr<Value> rhs = residual(conjunction.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    if (!*lhs)
      return *rhs;
    if (!*rhs)
      return *lhs;
    if (!(*lhs).getType().isInteger(1) || !(*rhs).getType().isInteger(1))
      return failure();
    return Value(builder.create<BinaryOp>(location, builder.getI1Type(), *lhs,
                                          *rhs,
                                          BinaryOperator::LogicalAnd));
  };

  FailureOr<Value> authorResidual = residual(original);
  if (failed(authorResidual))
    return failure();
  if (!*authorResidual)
    return physicalTail;
  if (!physicalTail)
    return projectPhysicalValueToSchema(builder, location, *authorResidual,
                                          target);
  return materializeValidityConjunction(builder, location, physicalTail,
                                        *authorResidual, target);
}

} // namespace intent::gpu
