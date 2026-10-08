#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_COMPUTE_STRICTARITHMETIC_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_COMPUTE_STRICTARITHMETIC_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cutile {
void legalizeStrictArithmetic(mlir::func::FuncOp kernel);
}

#endif
