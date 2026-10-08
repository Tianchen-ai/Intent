#include "AccessComposition.h"
#include "../Value/ReplayInsertion.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::gpu::access {

bool isZero(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  return integer && integer.getValue().isZero();
}

void bindUnchangedAccessValues(OpBuilder &builder, ValueRange values,
                               FragmentType source, FragmentType target,
                               ArrayRef<int64_t> slicedAxes, IRMapping &mapping) {
  Operation *owner = builder.getInsertionBlock()->getParentOp();
  auto kernel = dyn_cast<func::FuncOp>(owner);
  if (!kernel) kernel = owner->getParentOfType<func::FuncOp>();
  if (!kernel) return;
  DominanceInfo dominance(kernel);
  SmallVector<Value> pending(values.begin(), values.end());
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!value || !visited.insert(value).second || mapping.lookupOrNull(value)) continue;
    auto type = dyn_cast<FragmentType>(value.getType());
    if (!type) continue;
    auto from = queryBroadcastProjection(type, source);
    auto to = queryBroadcastProjection(type, target);
    bool untouched = from.isExact() && to.isExact() &&
        llvm::all_of(slicedAxes, [&](int64_t axis) {
          return axis >= 0 && static_cast<size_t>(axis) < from.targetToSource.size() &&
                 !from.targetToSource[axis];
        });
    if (untouched && availableAtInsertionPoint(value, builder, dominance)) {
      // A page/index load may have happened before a later metadata mutation.
      // Capture that old value exactly; the data load has its own read proof.
      mapping.map(value, value);
      continue;
    }
    Operation *producer = value.getDefiningOp();
    if (!producer || producer->getNumRegions() ||
        !isPhysicalReplayNode(producer, PhysicalReplayScope::Coordinate,
                              /*allowAccesses=*/false)) continue;
    llvm::append_range(pending, producer->getOperands());
  }
}

FailureOr<Value> replayFragmentValue(OpBuilder &builder, Value value,
                                     FragmentType target, IRMapping &mapping,
                                     PhysicalProgramAnalysis &analysis,
                                     Operation *insertionAnchor) {
  if (!value)
    return Value();
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return value;
  if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    Value replacement;
    for (const auto &entry : mapping.getValueMap()) {
      auto known = entry.first.getDefiningOp<MakeRangeOp>();
      if (!known || !sameLogicalRange(range, known) ||
          !samePhysicalScalarExpression(range.getStart(), known.getStart()) ||
          !samePhysicalScalarExpression(range.getExtent(), known.getExtent()))
        continue;
      if (replacement && replacement != entry.second)
        return failure();
      replacement = entry.second;
    }
    if (!replacement)
      return failure();
    mapping.map(value, replacement);
    return replacement;
  }
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
    if (!record)
      return failure();
    FailureOr<Value> field = replayFragmentValue(
        builder, record.getFields()[extract.getField()], target, mapping, analysis,
        insertionAnchor);
    if (failed(field))
      return failure();
    mapping.map(value, *field);
    return *field;
  }
  PhysicalReplayFact replay = analysis.replayAt(
      value, std::nullopt, PhysicalReplayScope::Coordinate,
      /*allowAccesses=*/false, insertionAnchor, mapping);
  Operation *producer = value.getDefiningOp();
  if (!producer || !replay.isReplayable() ||
      !isPhysicalReplayNode(producer, PhysicalReplayScope::Coordinate,
                            /*allowAccesses=*/false))
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replacement =
        replayFragmentValue(builder, operand, target, mapping, analysis,
                            insertionAnchor);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
  }
  if (isa<BroadcastOp, ReshapeOp, TransposeOp>(producer)) {
    auto resultType = FragmentType::get(
        target.getContext(), fragment.getElementType(), target.getShape(),
        target.getAxisMaps(), target.getValidity(), target.getOwner());
    // Replayed coordinates already use the destination axes.  The old shape
    // relation must not introduce its singleton axes a second time.
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, producer->getLoc(),
        mapping.lookupOrDefault(producer->getOperand(0)), resultType);
    if (failed(projected))
      return failure();
    mapping.map(value, *projected);
    return *projected;
  }
  SmallVector<Type> types;
  for (Type type : producer->getResultTypes()) {
    auto original = dyn_cast<FragmentType>(type);
    types.push_back(original ? Type(FragmentType::get(
        target.getContext(), original.getElementType(), target.getShape(),
        target.getAxisMaps(), target.getValidity(), target.getOwner())) : type);
  }
  if (failed(cloneWithSchema(builder, producer, mapping, types)))
    return failure();
  return mapping.lookup(value);
}

FailureOr<Value> replayScalarValue(OpBuilder &builder, Value value,
                                   IRMapping &mapping,
                                   PhysicalProgramAnalysis &analysis,
                                   Operation *insertionAnchor) {
  if (!value)
    return Value();
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return value;
  if (Operation *projection = value.getDefiningOp();
      projection && isa<BroadcastOp, ReshapeOp, TransposeOp, SplatOp>(projection)) {
    FailureOr<Value> scalar =
        replayScalarValue(builder, projection->getOperand(0), mapping, analysis,
                          insertionAnchor);
    if (succeeded(scalar))
      mapping.map(value, *scalar);
    return scalar;
  }
  PhysicalReplayFact replay = analysis.replayAt(
      value, std::nullopt, PhysicalReplayScope::Coordinate,
      /*allowAccesses=*/false, insertionAnchor, mapping);
  Operation *producer = value.getDefiningOp();
  if (!producer || isa<MakeRangeOp>(producer) || !replay.isReplayable() ||
      !isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer))
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replacement =
        replayScalarValue(builder, operand, mapping, analysis, insertionAnchor);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
  }
  SmallVector<Type> types;
  for (Type type : producer->getResultTypes()) {
    auto fragment = dyn_cast<FragmentType>(type);
    types.push_back(fragment ? fragment.getElementType() : type);
  }
  if (failed(cloneWithSchema(builder, producer, mapping, types)))
    return failure();
  return mapping.lookup(value);
}

FailureOr<Value> combinePredicates(OpBuilder &builder, Location location,
                                   Type valueType, Value lhs,
                                   Value rhs) {
  Type predicate = builder.getI1Type();
  if (auto fragment = dyn_cast<FragmentType>(valueType))
    predicate = FragmentType::get(
        fragment.getContext(), builder.getI1Type(), fragment.getShape(),
        fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
  for (Value *value : {&lhs, &rhs}) {
    if (!*value)
      continue;
    if ((*value).getType() != predicate) {
      FailureOr<Value> projected =
          projectPhysicalValueToSchema(builder, location, *value, predicate);
      if (failed(projected))
        return failure();
      *value = *projected;
    }
  }
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return Value(builder.create<BinaryOp>(location, predicate, lhs, rhs,
                                        BinaryOperator::LogicalAnd));
}

FailureOr<Value> gatherBounds(OpBuilder &builder, Location location,
                              FragmentType source, ValueRange coordinates,
                              ArrayRef<int64_t> axes, Type indexType) {
  Type predicate = builder.getI1Type();
  if (auto fragment = dyn_cast<FragmentType>(indexType))
    predicate = FragmentType::get(builder.getContext(), builder.getI1Type(),
        fragment.getShape(), fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  auto lower = projectPhysicalValueToSchema(builder, location, zero, indexType);
  if (failed(lower))
    return failure();
  Value valid;
  for (auto [coordinate, axis] : llvm::zip(coordinates, axes)) {
    auto projected = projectPhysicalValueToSchema(builder, location, coordinate, indexType);
    auto extent = cast<PhysicalExprAttr>(source.getShape()[axis]);
    Value bound = extent.getKind() == PhysicalExprKind::Constant
        ? Value(builder.create<arith::ConstantIndexOp>(location, extent.getValue()))
        : Value(builder.create<PhysicalExprOp>(location, builder.getIndexType(), extent));
    auto upper = projectPhysicalValueToSchema(builder, location, bound, indexType);
    if (failed(projected) || failed(upper))
      return failure();
    Value nonnegative = builder.create<CompareOp>(location, predicate,
        *projected, *lower, ComparePredicate::Ge);
    Value below = builder.create<CompareOp>(location, predicate,
        *projected, *upper, ComparePredicate::Lt);
    Value bounded = builder.create<BinaryOp>(location, predicate,
        nonnegative, below, BinaryOperator::LogicalAnd);
    valid = valid ? Value(builder.create<BinaryOp>(location, predicate,
        valid, bounded, BinaryOperator::LogicalAnd)) : bounded;
  }
  return valid;
}

bool sameUniformValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  UniformValueAnalysis constants(describeUniformValue);
  if (equalUniformConstants(constants.evaluate(lhs), constants.evaluate(rhs)))
    return true;
  auto scalar = [](Value value) {
    while (value) {
      UniformExpression expression = describeUniformValue(value);
      if (expression.kind != UniformKind::Forward || expression.operands.size() != 1)
        break;
      value = expression.operands.front();
    }
    return value;
  };
  lhs = scalar(lhs);
  rhs = scalar(rhs);
  if (!lhs || !rhs || isa<FragmentType>(lhs.getType()) ||
      isa<FragmentType>(rhs.getType()))
    return false;
  if (lhs == rhs)
    return true;
  return false;
}

bool unitAxes(FragmentType type, ArrayRef<unsigned> axes) {
  return llvm::all_of(axes, [&](unsigned axis) {
    auto extent = cast<PhysicalExprAttr>(type.getShape()[axis]);
    return extent.getKind() == PhysicalExprKind::Constant && extent.getValue() == 1;
  });
}

Value cancelIndexOffset(OpBuilder &builder, Value coordinate, Value offset) {
  if (!uniformElementType(coordinate.getType()).isIndex())
    return {};
  if (auto subtract = coordinate.getDefiningOp<BinaryOp>();
      subtract && subtract.getOperatorKind() == BinaryOperator::Subtract) {
    Value bound = subtract.getRhs();
    while (true) {
      UniformExpression expression = describeUniformValue(bound);
      if (expression.kind != UniformKind::Forward ||
          expression.operands.size() != 1)
        break;
      bound = expression.operands.front();
    }
    if (samePhysicalScalarExpression(bound, offset))
      return subtract.getLhs();
  }
  Operation *projection = coordinate.getDefiningOp();
  if (!isa_and_nonnull<BroadcastOp, ReshapeOp, TransposeOp>(projection))
    return {};
  Value translated = cancelIndexOffset(builder, projection->getOperand(0), offset);
  if (!translated)
    return {};
  IRMapping mapping;
  mapping.map(projection->getOperand(0), translated);
  return builder.clone(*projection, mapping)->getResult(0);
}

} // namespace intent::gpu::access
