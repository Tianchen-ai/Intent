#ifndef INTENT_TRANSFORMS_PASSES_H
#define INTENT_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace intent {

mlir::LogicalResult verifyKernelStructure(mlir::ModuleOp module);
mlir::LogicalResult verifyKernelModule(mlir::ModuleOp module);
std::unique_ptr<mlir::Pass> createVerifyKernelIRPass();
std::unique_ptr<mlir::Pass> createNormalizeKernelIRPass();
mlir::LogicalResult normalizeKernelModule(mlir::ModuleOp module);
void registerIntentPasses();

} // namespace intent

#endif
