#ifndef INTENT_CPU_TRANSFORMS_CONTROL_TRAVERSALS_H
#define INTENT_CPU_TRANSFORMS_CONTROL_TRAVERSALS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cpu {

// Share compatible sibling iteration spaces, then forward same-coordinate
// values within the resulting control scopes and release dead private storage.
// This complete buffer transformation uses current effects, coordinates and
// reduction permissions. It neither assigns implementations nor vectorizes;
// callers discard storage/dominance facts after it rewrites the function.
mlir::LogicalResult fuseSharedTraversals(mlir::func::FuncOp function);

} // namespace intent::cpu
#endif
