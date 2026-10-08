#ifndef INTENT_TARGET_TRITON_SERIALIZATION_SERIALIZER_H
#define INTENT_TARGET_TRITON_SERIALIZATION_SERIALIZER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace intent::triton {

mlir::LogicalResult verifySourceOperation(mlir::Operation *operation);

mlir::LogicalResult serializeProgram(mlir::ModuleOp module,
                                    std::string &source,
                                    std::string &metadata);

} // namespace intent::triton

#endif
