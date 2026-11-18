#ifndef INTENT_DSA_TRANSFORMS_COLLECTIVE_COLLECTIVES_H
#define INTENT_DSA_TRANSFORMS_COLLECTIVE_COLLECTIVES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::bangc::physical {

mlir::LogicalResult realizeCollectives(mlir::func::FuncOp function);
void realizeIndependentReductions(mlir::func::FuncOp function);

} // namespace intent::bangc::physical

#endif
