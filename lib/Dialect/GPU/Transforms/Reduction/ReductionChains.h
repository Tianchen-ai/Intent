#ifndef INTENT_GPU_TRANSFORMS_REDUCTIONCHAINS_H
#define INTENT_GPU_TRANSFORMS_REDUCTIONCHAINS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu::reduction {

bool scalarReductionCombine(mlir::Region &source, mlir::Region &result);
bool sameReductionContract(ReduceOp inner, ReduceOp outer,
                           mlir::Region &scalarOuterCombine);
// Join direct ordinary reductions before choosing their physical traversal.
// This changes only the axes of the same member/identity/callback contract.
bool combineNestedReductions(mlir::func::FuncOp kernel);

} // namespace intent::gpu::reduction

#endif
