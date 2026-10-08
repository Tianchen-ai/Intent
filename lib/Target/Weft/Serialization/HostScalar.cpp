#include "Intent/Target/Weft/Serialization/HostScalar.h"
#include "Intent/Serialization/ScalarEmitters.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
namespace intent::weft_provider {
std::optional<DenseStorageType> denseStorageType(StringRef family) {
  if (family == "f16") return DenseStorageType{family.str(), "_Float16", 2};
  if (family == "bf16") return DenseStorageType{family.str(), "uint16_t", 2};
  if (family == "f8E4M3FN" || family == "f8E5M2")
    return DenseStorageType{family.str(), "uint8_t", 1};
  if (family == "f32") return DenseStorageType{family.str(), "float", 4};
  if (family == "f64") return DenseStorageType{family.str(), "double", 8};
  StringRef width = family;
  bool isUnsigned = width.consume_front("u");
  if (!isUnsigned && !width.consume_front("i")) return std::nullopt;
  unsigned bits = 0;
  if (width.getAsInteger(10, bits) ||
      (bits != 1 && bits != 8 && bits != 16 && bits != 32 && bits != 64))
    return std::nullopt;
  unsigned storageBits = std::max(8u, bits);
  return DenseStorageType{family.str(),
      std::string(isUnsigned ? "uint" : "int") +
          std::to_string(storageBits) + "_t", storageBits / 8};
}

std::optional<DenseStorageType> denseStorageType(Type type) {
  if (type.isIndex()) return denseStorageType("i64");
  if (auto integer = dyn_cast<IntegerType>(type))
    return denseStorageType((Twine(integer.isUnsigned() ? "u" : "i") +
                             Twine(integer.getWidth())).str());
  if (!isa<FloatType>(type)) return std::nullopt;
  std::string family;
  llvm::raw_string_ostream stream(family);
  type.print(stream);
  return denseStorageType(family);
}

std::optional<CScalarType> hostStorageType(Type type) {
  auto storage = denseStorageType(type);
  if (!storage) return std::nullopt;
  return CScalarType{storage->cType, {}, {}};
}

std::optional<CScalarType> hostScalarType(Type type) {
  // BF16 memory is raw storage. Its numerical conversion belongs to the
  // canonical task and target lowering, not C integer arithmetic in the host.
  if (type.isBF16() || isa<Float8E4M3FNType, Float8E5M2Type>(type)) return std::nullopt;
  if (type.isInteger(1)) return CScalarType{"int", {}, {}};
  return hostStorageType(type);
}
namespace {
FailureOr<std::string> render(Operation *operation, ArrayRef<std::string> operands) {
  auto scalar = ScalarOperation::read(operation);
  if (failed(scalar)) return failure();
  return emitCScalar(*scalar, operands, hostScalarType);
}
const auto &emitters() {
  static const auto table = scalarExpressionEmitters(render);
  return table;
}
} // namespace

FailureOr<std::string> emitHostScalar(Operation *operation, ArrayRef<std::string> operands) {
  return emitScalarExpression(emitters(), operation, operands);
}
LogicalResult verifyHostScalar(Operation *operation) { return emitters().verify(operation); }

} // namespace intent::weft_provider
