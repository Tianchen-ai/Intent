#ifndef INTENT_CPU_TRANSFORMS_INTEGERSOURCES_H
#define INTENT_CPU_TRANSFORMS_INTEGERSOURCES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cpu {

// Replay pure integer element definitions at current-IR scalar reads before
// private storage reuse can merge their independent value lifetimes.
mlir::LogicalResult foldIntegerSources(mlir::func::FuncOp function);

} // namespace intent::cpu
#endif
