#ifndef INTENT_TARGET_BANGC_SERIALIZATION_SCALAR_H
#define INTENT_TARGET_BANGC_SERIALIZATION_SCALAR_H
#include "Intent/Serialization/Scalar.h"

namespace intent::bangc {
std::optional<CScalarType> scalarType(mlir::Type type);
mlir::FailureOr<std::string> emitScalar(mlir::Operation *operation,
                                      llvm::ArrayRef<std::string> operands);
mlir::LogicalResult verifyScalar(mlir::Operation *operation);
} // namespace intent::bangc
#endif
