#ifndef INTENT_TARGET_TRITON_TRANSFORMS_COLLECTIVE_COLLECTIVES_H
#define INTENT_TARGET_TRITON_TRANSFORMS_COLLECTIVE_COLLECTIVES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::triton::detail {

void foldIntegerScanTails(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeCollectiveCallbacks(mlir::func::FuncOp kernel);

} // namespace intent::triton::detail

#endif
