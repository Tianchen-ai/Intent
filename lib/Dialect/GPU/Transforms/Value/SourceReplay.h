#ifndef INTENT_GPU_TRANSFORMS_VALUE_SOURCEREPLAY_H
#define INTENT_GPU_TRANSFORMS_VALUE_SOURCEREPLAY_H

#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"

namespace intent::gpu {
class ReplayPolicy;

// Close saved read coverage before creating consumer coordinates. The original
// read remains the snapshot authority; only its pure users may be rebuilt.
mlir::LogicalResult prepareSourceReplay(mlir::Value value, unsigned axis,
    mlir::Operation *anchor, PhysicalReplayScope scope);

// Bind saved frontiers before issuing the consumer's remaining memory reads.
// The caller owns its access geometry and neutralization of the final source.
mlir::LogicalResult bindSourceReplay(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value root,
    PhysicalSourceAxis source, int64_t dimension, std::optional<unsigned> sourceAxis,
    PhysicalExprAttr extent, mlir::Value coordinates, mlir::Operation *anchor,
    PhysicalReplayScope scope, AxisMapAttr resultMapping, mlir::IRMapping &mapping,
    const ReplayPolicy *reuse = nullptr);

mlir::FailureOr<mlir::Value> materializeSourceValue(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    PhysicalSourceAxis source, PhysicalExprAttr extent, mlir::IRMapping &mapping,
    mlir::Operation *anchor, ReplayMaterializationOptions options,
    mlir::Attribute sourceTail, const ReplayPolicy &reuse);

mlir::FailureOr<mlir::Value> materializeSourceRanges(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    PhysicalExprAttr extent, llvm::ArrayRef<MakeRangeOp> roots,
    mlir::Value coordinates, mlir::IRMapping &mapping, mlir::Operation *anchor);
mlir::FailureOr<mlir::Value> materializeSourceRanges(
    mlir::OpBuilder &builder, mlir::Location location, mlir::func::FuncOp kernel,
    mlir::Value value, PhysicalSourceAxis source, PhysicalExprAttr extent,
    MakeRangeOp root, mlir::Value coordinates, mlir::IRMapping &mapping,
    mlir::Operation *anchor);

mlir::FailureOr<mlir::Value> materializeSourceRead(
    mlir::OpBuilder &builder, mlir::Location location, LoadOp load, unsigned axis,
    FragmentType resultType, mlir::ValueRange coordinates, mlir::Value valid,
    mlir::Value fill, mlir::Value traversalCoordinate, mlir::Operation *anchor);

} // namespace intent::gpu
#endif
