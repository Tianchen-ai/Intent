#ifndef INTENT_TARGET_TRITON_TRANSFORMS_ACCESS_GATHERS_H
#define INTENT_TARGET_TRITON_TRANSFORMS_ACCESS_GATHERS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace intent::triton::detail {

mlir::LogicalResult materializeGatherSources(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeLargeScalarGathers(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeGatherCoordinates(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeExpandingGathers(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeSplitGatherPairs(mlir::func::FuncOp kernel);

mlir::Value stripShapeOnly(mlir::Value value);
mlir::FailureOr<mlir::Value> zeroLike(mlir::OpBuilder &builder,
                                    mlir::Location location, mlir::Type type);
bool belongsToSplitGatherPair(gpu::GatherOp gather);

} // namespace intent::triton::detail

#endif
