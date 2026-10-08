#ifndef INTENT_GPU_TRANSFORMS_CONTROL_TRAVERSALTAILS_H
#define INTENT_GPU_TRANSFORMS_CONTROL_TRAVERSALTAILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

// Split an existing physical traversal at its last complete tile. The caller
// owns predicate simplification and relation/configuration closure afterwards.
bool peelTraversalTails(mlir::func::FuncOp kernel);

} // namespace intent::gpu

#endif
