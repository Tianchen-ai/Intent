#ifndef INTENT_DIALECT_GPU_ANALYSIS_WORKSPACE_H
#define INTENT_DIALECT_GPU_ANALYSIS_WORKSPACE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

// Rectangular allocation capacity for every current shared configuration.
// Substitute each complete row before taking per-axis maxima. Runtime ABI
// expressions and host-bound deferred coverage parameters remain symbolic.
mlir::FailureOr<mlir::ArrayAttr>
workspaceAllocationShape(mlir::func::FuncOp kernel, mlir::ArrayAttr shape);

} // namespace intent::gpu

#endif
