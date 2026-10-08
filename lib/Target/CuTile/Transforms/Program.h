#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_PROGRAM_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_PROGRAM_H

#include "mlir/IR/BuiltinOps.h"

namespace intent::cutile {

// Shared GPU input is verified and its buffers/workspaces are executable.
mlir::LogicalResult prepareProgram(mlir::ModuleOp module);
// Consume current-GPU facts before committing native replacements.
mlir::LogicalResult formNativeProgram(mlir::ModuleOp module);
// Finish native forms and verify the complete provider program.
mlir::LogicalResult finalizeProgram(mlir::ModuleOp module,
                                   bool hoistLoopInvariants);

} // namespace intent::cutile

#endif
