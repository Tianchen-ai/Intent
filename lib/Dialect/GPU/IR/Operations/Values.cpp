#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "Verification.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"

using namespace mlir;
namespace intent::gpu {
using namespace operation_detail;

namespace {

bool sameExecutionShape(Type lhs, Type rhs) {
  auto left = dyn_cast<FragmentType>(lhs);
  auto right = dyn_cast<FragmentType>(rhs);
  if (static_cast<bool>(left) != static_cast<bool>(right))
    return false;
  return !left || (left.getShape() == right.getShape() &&
                   left.getValidity() == right.getValidity() &&
                   left.getOwner() == right.getOwner());
}

LogicalResult verifyDataSchemas(Operation *operation, TypeRange operands,
                                Type result, bool resultIsPredicate = false) {
  for (auto [index, operand] : llvm::enumerate(operands))
    if (!sameShape(operand, result))
      {
        InFlightDiagnostic diagnostic = operation->emitOpError(
            "physical data shapes/ownership disagree: operand=");
        diagnostic << operand << ", result=" << result;
        if (Operation *definition =
                operation->getOperand(index).getDefiningOp())
          diagnostic << ", producer=" << *definition;
        return failure();
      }
  if (resultIsPredicate) {
    if (!elementType(result).isInteger(1))
      return operation->emitOpError("physical predicate result must be i1");
  } else {
    for (Type operand : operands)
      if (elementType(operand) != elementType(result))
        return operation->emitOpError("physical data element types disagree");
  }
  return success();
}

} // namespace

LogicalResult SplatOp::verify() {
  return getResult().getType().getElementType() == getValue().getType()
             ? success()
             : emitOpError("splat element/result type disagree");
}

LogicalResult BroadcastOp::verify() {
  if (auto input = dyn_cast<FragmentType>(getValue().getType())) {
    auto result = getResult().getType();
    if (input.getElementType() != result.getElementType() ||
        input.getOwner() != result.getOwner() ||
        input.getShape().size() > result.getShape().size())
      return emitOpError("broadcast physical schema is invalid")
             << "; input=" << input << "; result=" << result;
    auto relations = queryFragmentOperandRelations(getOperation());
    if (failed(relations) || !relations->front().hasCompatibleExtents())
      return emitOpError("broadcast physical axis projection is unknown or incompatible")
             << "; input=" << input << "; result=" << result;
  } else if (getValue().getType() != getResult().getType().getElementType()) {
      return emitOpError("scalar broadcast element type disagrees: value=")
             << getValue().getType()
             << ", result_element=" << getResult().getType().getElementType()
             << ", value_producer="
             << (getValue().getDefiningOp()
                     ? getValue().getDefiningOp()->getName().getStringRef()
                     : StringRef("block argument"))
             << ", result=" << getResult().getType();
  }
  return success();
}

LogicalResult UnaryOp::verify() {
  if (!sameShape(getInput().getType(), getResult().getType()) ||
      elementType(getInput().getType()) != elementType(getResult().getType()))
    return emitOpError("unary physical schema is invalid");
  return verifyPointwiseMathMode(getOperation(), elementType(getInput().getType()));
}

LogicalResult BinaryOp::verify() {
  if (getStrictRounding() &&
      (!elementType(getLhs().getType()).isF32() || getApproximate() ||
       getFlushToZero() ||
       (getOperatorKind() != BinaryOperator::Add &&
        getOperatorKind() != BinaryOperator::Subtract &&
        getOperatorKind() != BinaryOperator::Multiply)))
    return emitOpError("strict rounding requires non-approximate, non-FTZ f32 add/subtract/multiply");
  if (failed(verifyPointwiseMathMode(getOperation(), elementType(getLhs().getType()))))
    return failure();
  return verifyDataSchemas(getOperation(), {getLhs().getType(), getRhs().getType()},
                           getResult().getType());
}

LogicalResult CompareOp::verify() {
  if (!sameShape(getLhs().getType(), getRhs().getType()) ||
      !sameExecutionShape(getLhs().getType(), getResult().getType()) ||
      elementType(getLhs().getType()) != elementType(getRhs().getType()) ||
      !elementType(getResult().getType()).isInteger(1)) {
    return emitOpError("comparison physical schema is invalid: lhs=")
           << getLhs().getType() << ", rhs=" << getRhs().getType()
           << ", result=" << getResult().getType() << ", lhs_producer="
           << (getLhs().getDefiningOp()
                   ? getLhs().getDefiningOp()->getName().getStringRef()
                   : StringRef("block argument"))
           << ", rhs_producer="
           << (getRhs().getDefiningOp()
                   ? getRhs().getDefiningOp()->getName().getStringRef()
                   : StringRef("block argument"));
  }
  return success();
}

LogicalResult SelectOp::verify() {
  if (!sameShape(getCondition().getType(), getTrueValue().getType()) ||
      !sameShape(getTrueValue().getType(), getFalseValue().getType()) ||
      !sameShape(getTrueValue().getType(), getResult().getType()) ||
      !elementType(getCondition().getType()).isInteger(1) ||
      elementType(getTrueValue().getType()) != elementType(getFalseValue().getType()) ||
      elementType(getTrueValue().getType()) != elementType(getResult().getType()))
    return emitOpError("select physical schema is invalid");
  return success();
}

LogicalResult CastOp::verify() {
  if (sameShape(getValue().getType(), getResult().getType()))
    return success();
  return emitOpError("cast must preserve physical shape and ownership: ")
         << getValue().getType() << " vs " << getResult().getType();
}

LogicalResult BitcastOp::verify() {
  if (!sameShape(getValue().getType(), getResult().getType()))
    return emitOpError("bitcast must preserve physical shape and ownership");
  auto width = [](Type type) -> std::optional<unsigned> {
    if (auto integer = dyn_cast<IntegerType>(type))
      return integer.getWidth();
    if (auto floating = dyn_cast<FloatType>(type))
      return floating.getWidth();
    return std::nullopt;
  };
  auto source = width(elementType(getValue().getType()));
  auto target = width(elementType(getResult().getType()));
  return source && target && *source == *target
             ? success()
             : emitOpError("bitcast element widths disagree");
}

LogicalResult MakeRecordOp::verify() {
  auto result = getResult().getType();
  if (result.getFieldTypes().size() != getFields().size())
    return emitOpError("record fields do not match its physical type");
  for (auto [index, field, type] :
       llvm::enumerate(getFields(), result.getFieldTypes()))
    if (field.getType() != cast<TypeAttr>(type).getValue())
      return emitOpError("record field has the wrong physical type")
             << "; field=" << index << "; actual=" << field.getType()
             << "; expected=" << cast<TypeAttr>(type).getValue();
  return success();
}

LogicalResult ExtractOp::verify() {
  auto record = getRecord().getType();
  if (getField() >= record.getFieldTypes().size())
    return emitOpError("record projection is outside its physical schema")
           << "; field=" << getField()
           << "; field_count=" << record.getFieldTypes().size();
  Type expected =
      cast<TypeAttr>(record.getFieldTypes()[getField()]).getValue();
  if (getResult().getType() != expected)
    return emitOpError("record projection is outside its physical schema")
           << "; field=" << getField()
           << "; actual=" << getResult().getType()
           << "; expected=" << expected;
  return success();
}

LogicalResult RandomBitsOp::verify() {
  return elementType(getResult().getType()).isUnsignedInteger(32) &&
                 sameShape(getCounter().getType(), getResult().getType())
             ? success()
             : emitOpError("Philox result must be a shape-identical u32 value");
}

} // namespace intent::gpu
