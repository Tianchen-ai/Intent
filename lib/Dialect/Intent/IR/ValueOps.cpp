#include "TypeSchema.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace intent::detail;

namespace intent {

LogicalResult verifyPointwiseMathMode(Operation *operation, Type elementType) {
  auto approximate = operation->getAttrOfType<BoolAttr>("approximate");
  auto flush = operation->getAttrOfType<BoolAttr>("flush_to_zero");
  bool isApproximate = approximate && approximate.getValue();
  bool isFlush = flush && flush.getValue();
  if (!isApproximate && !isFlush)
    return success();
  if (!isApproximate || !elementType.isF32())
    return operation->emitOpError(
        "non-default math requires approximate=true and f32 operands/results");
  if (auto unary = operation->getAttrOfType<UnaryOperatorAttr>("operator_kind")) {
    if (unary.getValue() == UnaryOperator::Exp2 ||
        (unary.getValue() == UnaryOperator::Tanh && !isFlush))
      return success();
  } else if (auto binary =
                 operation->getAttrOfType<BinaryOperatorAttr>("operator_kind")) {
    if (binary.getValue() == BinaryOperator::TrueDivide)
      return success();
  }
  return operation->emitOpError(
      "math mode is defined only for exp2/division and non-FTZ tanh");
}

LogicalResult MakeTupleOp::verify() {
  Operation *operation = getOperation();
  auto result = dyn_cast<intent::TupleType>(getResult().getType());
  if (!result || result.getComponentTypes().size() != getComponents().size())
    return operation->emitOpError("tuple operands do not match result schema");
  for (auto [operand, typeAttribute] :
       llvm::zip(getComponents(), result.getComponentTypes()))
    if (operand.getType() != cast<TypeAttr>(typeAttribute).getValue())
      return operation->emitOpError("tuple component operand has the wrong type");
  return success();
}

LogicalResult MakeRecordOp::verify() {
  Operation *operation = getOperation();
  auto result = dyn_cast<RecordType>(getResult().getType());
  if (!result || result.getFieldTypes().size() != getFields().size())
    return operation->emitOpError("record operands do not match result schema");
  for (auto [operand, typeAttribute] :
       llvm::zip(getFields(), result.getFieldTypes()))
    if (operand.getType() != cast<TypeAttr>(typeAttribute).getValue())
      return operation->emitOpError("record field operand has the wrong type");
  return success();
}

LogicalResult ExtractOp::verify() {
  Operation *operation = getOperation();
  auto field = getFieldAttr();
  if (!field || field.getInt() < 0)
    return operation->emitOpError("product projection requires a valid component");
  if (auto tuple = dyn_cast<intent::TupleType>(getProduct().getType())) {
    if (static_cast<size_t>(field.getInt()) >= tuple.getComponentTypes().size())
      return operation->emitOpError(
          "tuple projection references an invalid component");
    if (getResult().getType() !=
        cast<TypeAttr>(tuple.getComponentTypes()[field.getInt()]).getValue())
      return operation->emitOpError("tuple projection result type is incorrect");
    return success();
  }
  auto record = dyn_cast<RecordType>(getProduct().getType());
  if (!record ||
      static_cast<size_t>(field.getInt()) >= record.getFieldTypes().size())
    return operation->emitOpError("record projection references an invalid field");
  if (getResult().getType() !=
      cast<TypeAttr>(record.getFieldTypes()[field.getInt()]).getValue())
    return operation->emitOpError("record projection result type is incorrect");
  return success();
}

LogicalResult UnaryOp::verify() {
  Operation *operation = getOperation();
  Type result = getResult().getType();
  Type input = getInput().getType();
  auto kind = getOperatorKindAttr();
  if (!kind || !sameDataSchema(input, result))
    return operation->emitOpError("unary operator/schema is invalid");
  if (failed(verifyPointwiseMathMode(operation, getElementType(input))))
    return failure();
  if (kind.getValue() == UnaryOperator::Not)
    return isBooleanData(input)
               ? success()
               : operation->emitOpError("logical not requires bool data");
  if (kind.getValue() == UnaryOperator::Negate ||
      kind.getValue() == UnaryOperator::Abs)
    return isNumericData(input)
               ? success()
               : operation->emitOpError("numeric unary requires numeric data");
  return isa<FloatType>(getElementType(input))
             ? success()
             : operation->emitOpError(
                   "transcendental unary operation requires floating data");
}

LogicalResult BinaryOp::verify() {
  Operation *operation = getOperation();
  Type result = getResult().getType();
  Type lhs = getLhs().getType();
  Type rhs = getRhs().getType();
  auto kind = getOperatorKindAttr();
  if (!kind || !sameDataSchema(lhs, rhs, false) ||
      !sameDataSchema(lhs, result, false))
    return operation->emitOpError("binary operator/schema is invalid");
  BinaryOperator value = kind.getValue();
  if (failed(verifyPointwiseMathMode(operation, getElementType(lhs))))
    return failure();
  if (value == BinaryOperator::LogicalAnd ||
      value == BinaryOperator::LogicalOr)
    return isBooleanData(lhs) && isBooleanData(rhs) && isBooleanData(result)
               ? success()
               : operation->emitOpError(
                     "logical binary operation requires bool data");
  if (value == BinaryOperator::BitwiseAnd ||
      value == BinaryOperator::BitwiseOr ||
      value == BinaryOperator::BitwiseXor ||
      value == BinaryOperator::LeftShift ||
      value == BinaryOperator::RightShift) {
    Type lhsElement = getElementType(lhs);
    Type rhsElement = getElementType(rhs);
    Type resultElement = getElementType(result);
    return isa<IntegerType, IndexType, LogicalIndexType>(lhsElement) &&
                   isa<IntegerType, IndexType, LogicalIndexType>(rhsElement) &&
                   isa<IntegerType, IndexType, LogicalIndexType>(resultElement)
               ? success()
               : operation->emitOpError(
                     "bitwise binary operation requires integer/index data");
  }
  if (!isNumericData(lhs) || !isNumericData(rhs) || !isNumericData(result))
    return operation->emitOpError(
        "arithmetic binary operation requires numeric data");
  if (value == BinaryOperator::TrueDivide &&
      !isa<FloatType>(getElementType(lhs)))
    return operation->emitOpError(
        "true division requires explicitly floating operands");
  if ((value == BinaryOperator::FloorDivide ||
       value == BinaryOperator::Remainder) &&
      !isa<IntegerType, IndexType, LogicalIndexType>(getElementType(lhs)))
    return operation->emitOpError(
        "floor division/remainder require integer/index operands");
  if ((value == BinaryOperator::MaximumNum ||
       value == BinaryOperator::MinimumNum) &&
      !isa<FloatType>(getElementType(lhs)))
    return operation->emitOpError(
        "NaN-selecting min/max require floating operands");
  return success();
}

LogicalResult CompareOp::verify() {
  Operation *operation = getOperation();
  Type result = getResult().getType();
  Type lhs = getLhs().getType();
  Type rhs = getRhs().getType();
  auto predicate = getPredicateAttr();
  if (!predicate || !sameDataSchema(lhs, rhs, false) ||
      !sameDataSchema(lhs, result, false) || !isBooleanData(result) ||
      (!sameDataSchema(lhs, rhs) &&
       !(isNumericData(lhs) && isNumericData(rhs))))
    return operation->emitOpError("comparison schema/predicate is invalid");
  return success();
}

namespace {

LogicalResult verifySelection(Operation *operation, Type condition, Type lhs,
                             Type rhs, Type result) {
  if (!isBooleanData(condition) || !sameDataSchema(condition, lhs, false) ||
      !sameDataSchema(lhs, rhs) || !sameDataSchema(lhs, result))
    return operation->emitOpError(
        "select/mask operands must already have one canonical broadcast schema");
  return success();
}

} // namespace

LogicalResult SelectOp::verify() {
  return verifySelection(*this, getCondition().getType(), getTrueValue().getType(),
                         getFalseValue().getType(), getResult().getType());
}

LogicalResult MaskOp::verify() {
  return verifySelection(*this, getPredicate().getType(), getValue().getType(),
                         getFill().getType(), getResult().getType());
}

LogicalResult CastOp::verify() {
  Operation *operation = getOperation();
  Type result = getResult().getType();
  Type input = getInput().getType();
  if (!sameDataSchema(input, result, false))
    return operation->emitOpError("cast must preserve logical shape");
  auto rounding = getRoundingAttr();
  if (rounding && rounding.getInt() != 0)
    return operation->emitOpError("cast rounding is outside the language enum");
  return success();
}

LogicalResult BitcastOp::verify() {
  Operation *operation = getOperation();
  Type result = getResult().getType();
  Type input = getInput().getType();
  if (!sameDataSchema(input, result, false))
    return operation->emitOpError("cast must preserve logical shape");
  auto width = [](Type type) -> std::optional<unsigned> {
    if (auto integer = dyn_cast<IntegerType>(type))
      return integer.getWidth();
    if (auto floating = dyn_cast<FloatType>(type))
      return floating.getWidth();
    return std::nullopt;
  };
  Type sourceElement = getElementType(input);
  Type resultElement = getElementType(result);
  auto sourceWidth = width(sourceElement);
  auto resultWidth = width(resultElement);
  if (!sourceWidth || !resultWidth || sourceElement.isInteger(1) ||
      resultElement.isInteger(1) || *sourceWidth != *resultWidth)
    return operation->emitOpError(
        "bitcast requires equal-width non-bool elements");
  return success();
}

LogicalResult ConstantOp::verify() {
  Operation *operation = getOperation();
  Attribute value = getValue();
  if (!value)
    return operation->emitOpError("constant requires a canonical value");
  Type result = getResult().getType();
  if ((isa<IntegerAttr>(value) && !isa<IntegerType, IndexType>(result)) ||
      (isa<FloatAttr>(value) && !isa<FloatType>(result)))
    return operation->emitOpError("constant attribute/result type mismatch");
  return success();
}

LogicalResult RandomBitsOp::verify() {
  Operation *operation = getOperation();
  Type counter = getLogicalCounter().getType();
  Type result = getResult().getType();
  if (!getSeed().getType().isUnsignedInteger(64) ||
      !isIntegerLike(getElementType(counter)) ||
      !getElementType(result).isUnsignedInteger(32))
    return operation->emitOpError(
        "Philox bits seed/counter/result schema is invalid");
  auto counterTensor = dyn_cast<RankedTensorType>(counter);
  auto resultTensor = dyn_cast<RankedTensorType>(result);
  if (static_cast<bool>(counterTensor) != static_cast<bool>(resultTensor) ||
      (counterTensor && !sameTensorShape(counterTensor, resultTensor)))
    return operation->emitOpError("Philox bits result must preserve counter shape");
  return success();
}

} // namespace intent
