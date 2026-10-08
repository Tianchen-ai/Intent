#ifndef INTENT_TARGET_TRITON_TRANSFORMS_ACCESS_SCATTER_H
#define INTENT_TARGET_TRITON_TRANSFORMS_ACCESS_SCATTER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::triton::detail {

mlir::LogicalResult legalizeScatterAdd(mlir::func::FuncOp kernel);

} // namespace intent::triton::detail

#endif
