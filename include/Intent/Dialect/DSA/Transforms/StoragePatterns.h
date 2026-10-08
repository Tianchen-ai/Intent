#ifndef INTENT_DSA_TRANSFORMS_STORAGE_PATTERNS_H
#define INTENT_DSA_TRANSFORMS_STORAGE_PATTERNS_H

#include "Intent/Dialect/DSA/Analysis/Storage.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace intent::dsa::detail {

// The operation defines every element of this operand, including padding.
// An accumulator read/write (matrix or collective state) is not an overwrite.
inline mlir::OpOperand *completeLocalOutput(mlir::Operation *operation) {
  if (auto op = mlir::dyn_cast<MatrixTileOp>(operation))
    return op.getAccumulate() ? nullptr : &op.getAccumulatorMutable();
  if (auto op = mlir::dyn_cast<ReduceOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<FillOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<LoadTileOp>(operation))
    return op.getAsynchronous() ? nullptr : &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<GatherRowsOp>(operation))
    return op.getAsynchronous() ? nullptr : &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<TransposeOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<CastOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<UnaryOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<BinaryOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<SelectOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<CompareOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<MaskedFillOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<IndexBinaryOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<IotaOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<DivideRNOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<DivideCastOp>(operation)) return &op.getOutputMutable();
  if (auto op = mlir::dyn_cast<mlir::memref::CopyOp>(operation)) return &op.getTargetMutable();
  return nullptr;
}

// These rewrites substitute coordinates directly. A common allocation alone
// does not preserve their indexing: both descriptors must expose the complete
// storage with the same element, shape and layout.
inline bool sameCompleteView(StorageAnalysis &storage, mlir::Value first,
                             mlir::Value second) {
  if (first == second)
    return true;
  if (first.getType() != second.getType())
    return false;
  mlir::Value origin = storage.uniqueOrigin(first);
  return origin && storage.uniqueOrigin(second) == origin &&
         isCompleteStorageViewOf(first, origin) &&
         isCompleteStorageViewOf(second, origin);
}

} // namespace intent::dsa::detail

#endif
