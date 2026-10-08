#ifndef INTENT_DIALECT_DSA_TRANSFORMS_PASSES_H
#define INTENT_DIALECT_DSA_TRANSFORMS_PASSES_H

#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir { class OpPassManager; }

namespace intent::dsa {
#define GEN_PASS_DECL
#include "Intent/Dialect/DSA/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "Intent/Dialect/DSA/Transforms/Passes.h.inc"

void buildDSAPipeline(mlir::OpPassManager &manager);
void registerDSAPipelines();
mlir::LogicalResult realizeCollectiveGatherSupply(mlir::func::FuncOp function);
mlir::LogicalResult realizeMatrixSupply(mlir::func::FuncOp function);
bool normalizeLinearIndices(mlir::func::FuncOp function);
void bindUniformOperands(mlir::func::FuncOp function);
bool reuseLocalMemoryValues(mlir::func::FuncOp function);
bool eliminateUnreadLocalWrites(mlir::func::FuncOp function);
bool composeLocalTransfers(mlir::func::FuncOp function);
bool forwardIndexExpressions(mlir::func::FuncOp function);
bool realizeRangeComparisons(mlir::func::FuncOp function);
bool foldRangeCounts(mlir::func::FuncOp function);
bool foldUniformLocalValues(mlir::func::FuncOp function);
bool reuseGatherOffsets(mlir::func::FuncOp function);
bool batchPointwiseTasks(mlir::func::FuncOp function);
} // namespace intent::dsa
#endif
