#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_BOUNDS_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_BOUNDS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::cutile {

mlir::FailureOr<llvm::SmallVector<mlir::Value>> uniformAlignmentFactors(
    mlir::Value value, mlir::Value divisor, unsigned depth = 0);
bool isAlignedPeriodicTile(mlir::Value start, mlir::Value extent,
                           mlir::Value period);
mlir::scf::ForOp completeAlignedTileLoop(mlir::Value start, mlir::Value extent);
bool constantOriginInView(int64_t origin, gpu::ViewType view,
                          unsigned resourceAxis, mlir::func::FuncOp kernel);
bool rangeOriginInView(gpu::MakeRangeOp range,
                       llvm::ArrayRef<mlir::Value> offsets, gpu::ViewType view,
                       unsigned resourceAxis, int64_t dimension,
                       mlir::func::FuncOp kernel);
bool scalarOriginInView(mlir::Value index, gpu::ViewType view,
                        unsigned resourceAxis, int64_t dimension,
                        mlir::func::FuncOp kernel);
bool scalarCoordinatesInView(mlir::ValueRange coordinates, gpu::ViewType view,
                              mlir::func::FuncOp kernel);

} // namespace intent::cutile

#endif
