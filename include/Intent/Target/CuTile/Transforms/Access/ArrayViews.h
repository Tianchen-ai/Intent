#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_ARRAYVIEWS_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_ARRAYVIEWS_H

#include "mlir/IR/BuiltinOps.h"

namespace intent::cutile {

mlir::LogicalResult collapseArrayViews(mlir::ModuleOp module);

} // namespace intent::cutile

#endif
