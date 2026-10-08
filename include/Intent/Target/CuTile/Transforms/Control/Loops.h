#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_CONTROL_LOOPS_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_CONTROL_LOOPS_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intent::cutile {
// Materialize native i32 iteration only when its bounds are proven; retain
// explicit wide induction otherwise. Both forms remain actual structured IR.
void realizeWideLoops(mlir::func::FuncOp kernel);
void preserveNativeIndexValues(mlir::func::FuncOp kernel);
}
#endif
