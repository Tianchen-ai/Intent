#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_TILEINDICES_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_TILEINDICES_H

#include "TilePlan.h"
#include "mlir/IR/Builders.h"

namespace intent::cutile {

struct MaterializedTileIndices {
  llvm::SmallVector<mlir::Value> values;
  mlir::Value alignment;
};

mlir::FailureOr<mlir::Value> tileIndex(mlir::OpBuilder &builder,
                                     mlir::Location location,
                                     mlir::Value start, mlir::Value extent);
mlir::FailureOr<MaterializedTileIndices> materializeTileIndices(
    mlir::OpBuilder &builder, mlir::Operation *owner,
    const NativeTileAccessPlan &plan, bool allowDynamicAlignment);
mlir::FailureOr<mlir::Value> materializeTileOriginGuard(
    mlir::OpBuilder &builder, mlir::Operation *owner, mlir::Value resource,
    NativeTileAccessPlan &plan, const gpu::PhysicalAccessBoundaryFact &boundary);
mlir::Value materializeFullRangeGuard(
    mlir::OpBuilder &builder, mlir::Location location,
    const gpu::PhysicalAccessBoundaryFact &boundary);
mlir::Value materializeFullTileCondition(mlir::OpBuilder &builder,
                                        mlir::Location location,
                                        mlir::Value resource,
                                        gpu::FragmentType tile);

} // namespace intent::cutile

#endif
