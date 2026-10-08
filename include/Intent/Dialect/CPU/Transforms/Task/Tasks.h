#ifndef INTENT_DIALECT_CPU_TRANSFORMS_TASK_TASKS_H
#define INTENT_DIALECT_CPU_TRANSFORMS_TASK_TASKS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

namespace intent::cpu {

class ImplementationRegistry;

mlir::LogicalResult groupWorksetComputations(
    mlir::func::FuncOp function, const ImplementationRegistry &implementations);
mlir::LogicalResult exposeStructuredWorksets(
    mlir::func::FuncOp function, const ImplementationRegistry &implementations,
    llvm::ArrayRef<mlir::Value> leadingExtents = {});
mlir::LogicalResult blockStructuredComputations(
    mlir::func::FuncOp function, const ImplementationRegistry &implementations);
// Form worker intervals for one independent root workset, preserving its grain
// assignment and the complete computation at each original logical point.
mlir::LogicalResult partitionWorkset(mlir::scf::ParallelOp workset, int64_t grain);
mlir::LogicalResult partitionTasks(
    mlir::func::FuncOp function, int64_t grain,
    const ImplementationRegistry &implementations);
mlir::LogicalResult isolateTasks(mlir::func::FuncOp function);
mlir::LogicalResult materializeTaskDispatches(mlir::func::FuncOp function);
// Keep constants local to outlined callbacks after the final canonicalization.
void localizeTaskConstants(mlir::func::FuncOp function);

} // namespace intent::cpu
#endif
