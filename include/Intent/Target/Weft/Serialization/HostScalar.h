#ifndef INTENT_TARGET_WEFT_SERIALIZATION_HOSTSCALAR_H
#define INTENT_TARGET_WEFT_SERIALIZATION_HOSTSCALAR_H
#include "Intent/Serialization/Scalar.h"

namespace intent::weft_provider {
struct DenseStorageType {
  std::string family;
  std::string cType;
  int64_t bytes;
};
std::optional<DenseStorageType> denseStorageType(llvm::StringRef family);
std::optional<DenseStorageType> denseStorageType(mlir::Type type);
std::optional<CScalarType> hostStorageType(mlir::Type type);
std::optional<CScalarType> hostScalarType(mlir::Type type);
mlir::FailureOr<std::string> emitHostScalar(mlir::Operation *operation,
                                          llvm::ArrayRef<std::string> operands);
mlir::LogicalResult verifyHostScalar(mlir::Operation *operation);
} // namespace intent::weft_provider
#endif
