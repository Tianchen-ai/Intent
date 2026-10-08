#ifndef INTENT_DIALECT_GPU_TRANSFORMS_TRAVERSAL_H
#define INTENT_DIALECT_GPU_TRANSFORMS_TRAVERSAL_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {

// Attach the loop to current IR before building a body that queries dominance,
// resource snapshots or enclosing program facts. The callback owns its yield.
mlir::scf::ForOp createTraversalLoop(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value lower,
    mlir::Value upper, mlir::Value step, mlir::ValueRange initialValues,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::Value,
                            mlir::ValueRange)> buildBody);

mlir::FailureOr<PhysicalExprAttr> boundedTraversalChunk(ParameterAttr chunk,
                                                       MakeRangeOp range);
// Clip a complete ownership domain using the original parameter's candidate
// domain, including its lower bound. Host-dependent capacities use the existing
// early Coverage binding; device-dependent or candidate-dependent bounds retain
// the original extent.
mlir::FailureOr<PhysicalExprAttr>
boundedOwnershipExtent(mlir::func::FuncOp kernel, ParameterAttr parameter,
                       PhysicalExprAttr logicalCapacity);
mlir::LogicalResult bindFullCoverageDimension(mlir::func::FuncOp kernel,
                                              uint64_t dimension,
                                              mlir::Value physicalExtent);
mlir::LogicalResult realizeFullCoverageDimension(mlir::func::FuncOp kernel,
                                                 mlir::Value source,
                                                 unsigned fragmentAxis);

} // namespace intent::gpu

#endif
