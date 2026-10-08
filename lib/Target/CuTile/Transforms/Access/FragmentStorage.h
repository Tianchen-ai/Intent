#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_FRAGMENTSTORAGE_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_FRAGMENTSTORAGE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cutile {
mlir::LogicalResult materializeFragmentStorage(mlir::func::FuncOp kernel);
}

#endif
