#ifndef INTENT_TARGET_TRITON_ANALYSIS_PROGRAM_H
#define INTENT_TARGET_TRITON_ANALYSIS_PROGRAM_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::triton {

// Validate current provider IR and its terminal source surface in one walk.
// The source registry is supplied by the caller, keeping analysis independent
// of source emission and transformation libraries.
mlir::LogicalResult verifyTritonProgram(
    mlir::ModuleOp module,
    llvm::function_ref<mlir::LogicalResult(mlir::Operation *)> verifySource);

} // namespace intent::triton

#endif
