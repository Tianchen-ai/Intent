#pragma once

#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace intent::cpu {

// A pointwise observation of a complete private accumulator. The proof belongs
// to the current IR snapshot; move the recorded pure dependencies before any
// contraction rewriting, then emit only after a tile's complete K traversal.
struct ContractionEpilogue {
  mlir::linalg::GenericOp consumer;
  unsigned accumulatorInput;
  llvm::SmallVector<mlir::Operation *> dependencies;
};

std::optional<ContractionEpilogue> queryContractionEpilogue(
    mlir::linalg::GenericOp contraction, mlir::Operation *initialization,
    StorageAnalysis &storage);

// Narrower than moving the observation: the entire old allocation must be
// removable, with only logical dimension queries left after computation moves.
bool canCompactContractionAccumulator(mlir::linalg::GenericOp contraction,
    mlir::Operation *initialization, ContractionEpilogue &epilogue,
    StorageAnalysis &storage);
// Logical dimensions retain their old values; a replacement allocation keeps
// the old release points instead of inheriting a new lifetime.
mlir::LogicalResult eraseContractionAccumulator(
    mlir::Value accumulator, mlir::Value replacementStorage = {});

mlir::LogicalResult emitContractionEpilogue(mlir::OpBuilder &builder,
    ContractionEpilogue &epilogue, mlir::ValueRange offsets,
    mlir::ValueRange sizes, mlir::Value completedTile = {});

} // namespace intent::cpu
