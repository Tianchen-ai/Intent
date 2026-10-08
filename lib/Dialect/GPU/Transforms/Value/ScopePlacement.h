#ifndef INTENT_GPU_TRANSFORMS_VALUE_SCOPEPLACEMENT_H
#define INTENT_GPU_TRANSFORMS_VALUE_SCOPEPLACEMENT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {
class ResourceAliasAnalysis;

namespace placement {

// These queries concern one current rewrite, not a cached schedule. Unknown
// effects and ordered accesses cannot establish independent execution.
bool independentMemoryEffects(mlir::Operation *first, mlir::Operation *second,
                              ResourceAliasAnalysis &aliases);

bool isMovableValueOperation(mlir::Operation *operation);
bool canMoveBefore(mlir::Operation *operation, mlir::Operation *before);

// Move only the intervening definitions required by second's operands and
// captures. No operation moves until the complete dependency slice is legal.
bool moveInputsBefore(mlir::Operation *first, mlir::Operation *second,
                      mlir::func::FuncOp kernel);

// Preserve ordinary read snapshots across a loop only under its actual memory
// independence and nonempty conditions. Pure invariant dependencies share the
// same bounded placement; provider layout and final resource checks remain.
mlir::LogicalResult hoistLoopInvariantValues(mlir::func::FuncOp kernel);

// An exact current control path, excluding the enclosing scope itself. Loops
// and unknown regions are not interchangeable with conditional execution.
mlir::FailureOr<llvm::SmallVector<mlir::Block *>>
conditionalPath(mlir::Block *scope, mlir::Operation *operation);

// Check only the executed lexical prefixes on that conditional path. This
// excludes the operation itself, not other effects that will also be relocated.
bool canMoveEffectBefore(mlir::Operation *operation, mlir::Operation *before,
                         ResourceAliasAnalysis &aliases);

// Rebuild the selected then/else path at one insertion scope. Successive bodies
// in original program order share their actual reconstructed ancestor branches.
// The caller proves memory motion and materializes conditions under the already
// reconstructed guards; no branch-local work is speculated outside its guard.
class ConditionalPlacement {
public:
  mlir::LogicalResult emit(
      mlir::OpBuilder &builder, llvm::ArrayRef<mlir::Block *> path,
      llvm::function_ref<mlir::FailureOr<mlir::Value>(
          mlir::OpBuilder &, mlir::Operation *)> condition,
      llvm::function_ref<mlir::LogicalResult(mlir::OpBuilder &)> body);

private:
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> branches;
};

} // namespace placement
} // namespace intent::gpu

#endif
