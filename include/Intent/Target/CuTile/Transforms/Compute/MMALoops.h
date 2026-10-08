#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_COMPUTE_MMALOOPS_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_COMPUTE_MMALOOPS_H

#include "mlir/IR/BuiltinOps.h"

namespace intent::cutile {

mlir::LogicalResult refineMMALoops(mlir::ModuleOp module);

} // namespace intent::cutile

#endif
