#ifndef INTENT_CPU_TRANSFORMS_STRUCTURE_REDUCTIONSUPPLY_H
#define INTENT_CPU_TRANSFORMS_STRUCTURE_REDUCTIONSUPPLY_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::cpu {

bool hasReductionSupplyGroup(mlir::linalg::GenericOp producer);

// The selected implementation supplies the callback and its actual SIMD width.
// The callback consumes complete source members; partial-state combine remains
// the original collective's responsibility.
mlir::FailureOr<bool> materializeReductionSupplyGroup(
    mlir::linalg::GenericOp producer, int64_t width,
    llvm::function_ref<mlir::LogicalResult(mlir::linalg::GenericOp,
        mlir::Value, llvm::ArrayRef<mlir::Operation *>)> materialize);

} // namespace intent::cpu

#endif
