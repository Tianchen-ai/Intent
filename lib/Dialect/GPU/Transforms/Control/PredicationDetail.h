#ifndef INTENT_GPU_TRANSFORMS_PREDICATIONDETAIL_H
#define INTENT_GPU_TRANSFORMS_PREDICATIONDETAIL_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/IR/Builders.h"

namespace intent::gpu::predication {

bool isFloatToIntegerCast(CastOp operation);
mlir::FailureOr<mlir::Value> selectScalarProduct(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value condition,
    mlir::Value lhs, mlir::Value rhs);
mlir::Value anyActiveLane(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value predicate);

} // namespace intent::gpu::predication
#endif
