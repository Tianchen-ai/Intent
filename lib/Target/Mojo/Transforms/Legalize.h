#ifndef INTENT_TARGET_MOJO_TRANSFORMS_LEGALIZE_H
#define INTENT_TARGET_MOJO_TRANSFORMS_LEGALIZE_H

#include "mlir/IR/BuiltinOps.h"

namespace intent::mojo {

// Complete provider transformations. Their inputs are the current CPU program;
// candidate and implementation bindings are already materialized in that IR.
mlir::LogicalResult prepareNativeProgram(mlir::ModuleOp module);
mlir::LogicalResult fusePrivateComputations(mlir::ModuleOp module);
mlir::LogicalResult vectorizeNativeProgram(mlir::ModuleOp module,
                                         bool fuseSharedTraversals);
mlir::LogicalResult finalizeNativeProgram(mlir::ModuleOp module);

} // namespace intent::mojo
#endif
