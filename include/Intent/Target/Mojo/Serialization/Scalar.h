#ifndef INTENT_TARGET_MOJO_SERIALIZATION_SCALAR_H
#define INTENT_TARGET_MOJO_SERIALIZATION_SCALAR_H

#include "Intent/Serialization/Scalar.h"

namespace intent::mojo {
bool supportsScalarType(mlir::Type type);
std::string scalarDType(mlir::Type type);
std::string scalarValueType(mlir::Type type);
mlir::FailureOr<std::string> emitScalar(mlir::Operation *operation,
                                      llvm::ArrayRef<std::string> operands);
mlir::LogicalResult verifyScalar(mlir::Operation *operation);
} // namespace intent::mojo
#endif
