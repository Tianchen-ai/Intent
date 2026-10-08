#ifndef INTENT_CPU_TRANSFORMS_CONTIGUOUSACCESSES_H
#define INTENT_CPU_TRANSFORMS_CONTIGUOUSACCESSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cpu {

// Compose full-domain accesses into typed views, retaining the logical rank and
// element order. Output forwarding remains owned by forwardCPUOutputs.
void foldContiguousAccesses(mlir::func::FuncOp function);

} // namespace intent::cpu
#endif
