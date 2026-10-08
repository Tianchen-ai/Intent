#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Intent/Dialect/CPU/Analysis/ViewRelations.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::weft_provider {

bool isAxisView(mlir::Operation *operation);
mlir::Value axisViewSource(mlir::Operation *operation);
bool isStaticShapeView(mlir::Operation *operation);
mlir::FailureOr<cpu::ViewAxisProjection> queryAxisView(mlir::Value value);

// The native task ABI represents contiguous storage, not arbitrary memref
// descriptors. Keep pure view definitions inside the task and capture their
// current contiguous descriptors and scalar operands before taking provider
// analysis snapshots. A forwarded descriptor remains the actual selected SSA.
mlir::LogicalResult reifyTaskViewCaptures(mlir::func::FuncOp function);

} // namespace intent::weft_provider
