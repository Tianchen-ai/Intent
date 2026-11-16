#ifndef INTENT_Block_TRANSFORMS_COMPLETEDREDUCTIONS_H
#define INTENT_Block_TRANSFORMS_COMPLETEDREDUCTIONS_H

#include "ReductionAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::block::reduction {

// A completed lane-wise partial, proved from the current init/update/member
// graph. The members are query inputs only; they are never replayed or moved.
struct CompletedReduction {
  mlir::scf::ForOp loop;
  llvm::SmallVector<mlir::Value> members;
  unsigned axis;
};

std::optional<CompletedReduction> queryCompletedReduction(ReduceOp reduce);

} // namespace intent::block::reduction
#endif
