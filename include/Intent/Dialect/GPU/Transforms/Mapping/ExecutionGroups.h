#ifndef INTENT_DIALECT_GPU_TRANSFORMS_EXECUTIONGROUPS_H
#define INTENT_DIALECT_GPU_TRANSFORMS_EXECUTIONGROUPS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent::gpu {

ExecutionGroupOp createExecutionGroup(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value linear,
    mlir::ValueRange extents, mlir::ArrayAttr launchExtents,
    mlir::DenseI64ArrayAttr roles, int64_t identity,
    PhysicalExprAttr offset, PhysicalExprAttr length);

// Transfer the actual execution body. The caller rebinds the old coordinate
// arguments and erases the emptied old group once all uses have been replaced.
ExecutionGroupOp rebuildExecutionGroup(
    mlir::OpBuilder &builder, ExecutionGroupOp group, mlir::ValueRange extents,
    mlir::ArrayAttr launchExtents, mlir::DenseI64ArrayAttr roles,
    PhysicalExprAttr length);

// Provider entry closes the shared execution scope into ordinary structured
// control and pure coordinates. No mapping facts survive on DelinearizeOp.
mlir::LogicalResult lowerExecutionGroups(mlir::ModuleOp module);

} // namespace intent::gpu
#endif
