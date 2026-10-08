#ifndef INTENT_TARGET_TRITON_TRANSFORMS_PROGRAM_H
#define INTENT_TARGET_TRITON_TRANSFORMS_PROGRAM_H

#include "mlir/IR/BuiltinOps.h"

namespace intent::triton {

// Complete provider phases consume and update the current executable program.
mlir::LogicalResult prepareTritonMemory(mlir::ModuleOp module);
mlir::LogicalResult formTritonProgram(mlir::ModuleOp module);
mlir::LogicalResult finalizeTritonProgram(mlir::ModuleOp module,
                                        bool hoistLoopInvariants);

} // namespace intent::triton

#endif
