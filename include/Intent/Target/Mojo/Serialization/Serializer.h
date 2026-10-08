#ifndef INTENT_TARGET_MOJO_SERIALIZATION_SERIALIZER_H
#define INTENT_TARGET_MOJO_SERIALIZATION_SERIALIZER_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace intent::mojo {
mlir::LogicalResult verifySourceOperation(mlir::Operation *operation);
mlir::LogicalResult verifySourceProgram(mlir::ModuleOp module);
mlir::LogicalResult serializeProgram(mlir::ModuleOp module, std::string &source,
                                    std::string &metadata);
}

#endif
