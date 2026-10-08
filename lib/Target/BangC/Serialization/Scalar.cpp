#include "Scalar.h"
#include "Intent/Serialization/ScalarEmitters.h"

using namespace mlir;
namespace intent::bangc {
std::optional<CScalarType> scalarType(Type type) {
  if (type.isF16()) return CScalarType{"half", {}, {}};
  if (type.isBF16())
    return CScalarType{"uint16_t", "intent_bf16_to_float", "intent_number_to_bf16"};
  if (type.isF32()) return CScalarType{"float", {}, {}};
  if (type.isF64()) return CScalarType{"double", {}, {}};
  if (isa<Float8E4M3FNType>(type))
    return CScalarType{"uint8_t", "intent_fp8_to_float<4, 3, 7, true>",
                       "intent_number_to_fp8<4, 3, 7, true>"};
  if (isa<Float8E5M2Type>(type))
    return CScalarType{"uint8_t", "intent_fp8_to_float<5, 2, 15, false>",
                       "intent_number_to_fp8<5, 2, 15, false>"};
  if (type.isInteger(1)) return CScalarType{"bool", {}, {}};
  if (type.isIndex()) return CScalarType{"int64_t", {}, {}};
  auto integer = dyn_cast<IntegerType>(type);
  if (!integer || !integer.isSignless() ||
      (integer.getWidth() != 8 && integer.getWidth() != 16 &&
       integer.getWidth() != 32 && integer.getWidth() != 64))
    return std::nullopt;
  return CScalarType{"int" + std::to_string(integer.getWidth()) + "_t", {}, {}};
}

namespace {
FailureOr<std::string> render(Operation *operation, ArrayRef<std::string> operands) {
  auto scalar = ScalarOperation::read(operation);
  if (failed(scalar)) return failure();
  if (scalar->kind == ScalarKind::Bitcast)
    return "intent_scalar_bitcast<" + scalarType(scalar->type())->name +
           ">(" + operands.front() + ")";
  if (scalar->kind == ScalarKind::Erf || scalar->kind == ScalarKind::Power ||
      scalar->kind == ScalarKind::Fma)
    return operation->emitError("scalar math primitive has no bound BANG C implementation"), failure();
  return emitCScalar(*scalar, operands, scalarType);
}
const auto &emitters() {
  static const auto table = scalarExpressionEmitters(render);
  return table;
}
} // namespace

FailureOr<std::string> emitScalar(Operation *operation, ArrayRef<std::string> operands) {
  return emitScalarExpression(emitters(), operation, operands);
}
LogicalResult verifyScalar(Operation *operation) { return emitters().verify(operation); }
} // namespace intent::bangc
