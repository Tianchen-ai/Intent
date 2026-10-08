#ifndef INTENT_DIALECT_CPU_TRANSFORMS_COLLECTIVE_COLLECTIVES_H
#define INTENT_DIALECT_CPU_TRANSFORMS_COLLECTIVE_COLLECTIVES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cpu {

mlir::LogicalResult realizeSliceCollectives(mlir::func::FuncOp function);
mlir::LogicalResult normalizeScalarReductions(mlir::func::FuncOp function);
// Outside existing parallel scopes, partial groups consume the function's
// bound task grain and module capabilities.
mlir::LogicalResult realizeHistograms(mlir::func::FuncOp function);

} // namespace intent::cpu
#endif
