#ifndef INTENT_Block_TRANSFORMS_REDUCTIONCHAINS_H
#define INTENT_Block_TRANSFORMS_REDUCTIONCHAINS_H

#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::block::reduction {

bool scalarReductionCombine(mlir::Region &source, mlir::Region &result);
bool sameReductionContract(ReduceOp inner, ReduceOp outer,
                           mlir::Region &scalarOuterCombine);
// Join direct ordinary reductions before choosing their physical traversal.
// This changes only the axes of the same member/identity/callback contract.
// After physicalization, fullOnly retains native scalar trees without
// reintroducing partially reduced axes that would need another traversal.
bool combineNestedReductions(mlir::func::FuncOp kernel, bool fullOnly = false);

} // namespace intent::block::reduction

#endif
