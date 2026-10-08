#ifndef INTENT_DIALECT_CPU_TRANSFORMS_STRUCTURE_COMPUTATIONS_H
#define INTENT_DIALECT_CPU_TRANSFORMS_STRUCTURE_COMPUTATIONS_H

#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::cpu {

mlir::LogicalResult fuseStructuredComputations(mlir::func::FuncOp function);
mlir::LogicalResult foldUniformComputations(mlir::func::FuncOp function);
mlir::LogicalResult materializeStructuredComputations(
    mlir::func::FuncOp function,
    llvm::function_ref<mlir::LogicalResult(mlir::Operation *)> materialize);
mlir::LogicalResult materializeStructuredComputation(
    mlir::Operation *operation, int64_t width, ImplementationAttr loopBinding);

} // namespace intent::cpu
#endif
