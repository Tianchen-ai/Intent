#ifndef INTENT_CPU_TRANSFORMS_STRUCTURE_PARALLELREDUCTIONS_H
#define INTENT_CPU_TRANSFORMS_STRUCTURE_PARALLELREDUCTIONS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include <optional>

namespace intent::cpu {

struct WorksetRows;

// A contiguous output coordinate selects independent states. Reduction axes
// keep their original relative order; lanes never participate in a horizontal
// reduction. The proof includes the current buffer aliases and scalar payload.
struct ParallelReduction {
  unsigned freeAxis;
  llvm::SmallVector<unsigned> reductionAxes;
  llvm::SmallVector<mlir::Value> guardedMemories;
};

std::optional<ParallelReduction>
queryParallelReduction(mlir::linalg::GenericOp operation);

mlir::LogicalResult orientParallelReductions(mlir::func::FuncOp function);

// Shared by workset exposure and materialization after the independent-state
// proof. The bound uses the current free extent and private state capacity.
WorksetRows parallelReductionWorkset(mlir::linalg::GenericOp operation,
                                    int64_t width, int64_t replicas);

// Invoked by implementations that already consume standard vector/SCF IR.
// Other implementations retain the shaped Generic and their native reduction.
mlir::FailureOr<bool> materializeParallelReduction(
    mlir::linalg::GenericOp operation, int64_t width, int64_t replicas,
    mlir::OpBuilder::Listener *listener, mlir::Value completedSupply = {},
    llvm::ArrayRef<mlir::Operation *> supplyReaders = {});

} // namespace intent::cpu

#endif
