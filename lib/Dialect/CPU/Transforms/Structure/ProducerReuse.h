#ifndef INTENT_CPU_TRANSFORMS_STRUCTURE_PRODUCERREUSE_H
#define INTENT_CPU_TRANSFORMS_STRUCTURE_PRODUCERREUSE_H

#include "Intent/Dialect/CPU/Analysis/ProducerReplay.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace intent::cpu {

// These queries select a rewrite after ProducerReplay has established legality.
// They describe this IR snapshot only, and never become execution metadata.
bool isLoadReplacement(const ProducerReplay &payload, mlir::Value result);
// Identify representation-only reconstruction; callers separately establish
// traversal geometry and the number of reads replacing the stored version.
mlir::Value convertedLoadSource(const ProducerReplay &payload, mlir::Value result);
// A profitability heuristic, unlike the zero-work/load-substitution case:
// inexpensive address arithmetic may be rebuilt in place of a temporary load.
// Integer-typed data computations and element reads do not qualify by dtype.
bool isRebuildableCoordinate(const ProducerReplay &payload, mlir::Value result);
mlir::FailureOr<ProducerReplay>
analyzeProducerResult(mlir::linalg::GenericOp producer, StorageAnalysis &storage,
                      unsigned resultNumber = 0);
// Invert a complete output projection. Additional memory axes must be zero
// coordinates of proven unit extents; every iteration dimension occurs once.
// The caller separately proves that its iteration bounds cover these extents.
mlir::FailureOr<mlir::AffineMap>
fullOutputProjection(mlir::Value output, mlir::AffineMap outputMap);
bool isNonRepeatingProjection(mlir::AffineMap coordinates,
                              llvm::ArrayRef<int64_t> iterationExtents);
// A complete-version rewrite supplies its actual number of consumer traversals.
// Repeating data computation is retained; simple representation reconstruction
// can trade conversions for eliminating wider stored elements and their reads.
bool shouldFuseProducerResult(const ProducerReplay &payload, mlir::Value result,
                              mlir::AffineMap coordinates,
                              llvm::ArrayRef<int64_t> iterationExtents,
                              bool removesProducer,
                              unsigned consumerTraversals = 1);
bool isRemovableProducerMetadata(mlir::Operation *operation);

struct ProducerReplayUse {
  mlir::Operation *operation;
  llvm::SmallVector<mlir::Value> coordinates;
  int64_t vectorWidth = 1;
  // Existing full-vector access legality may establish a tighter exclusive
  // end than an independent interval can express (full blocks before a tail).
  mlir::Value vectorLimit;
};

struct ProducerReplayGroup {
  mlir::Operation *anchor;
  llvm::SmallVector<mlir::Value> coordinates;
  llvm::SmallVector<mlir::Operation *> uses;
  int64_t vectorWidth;
  mlir::Value vectorLimit;
};

// Group equal coordinates at a real dominating insertion point. General data
// producers must disappear without repeating their original coordinate domain.
// Load substitution and the explicit coordinate-arithmetic heuristic permit
// per-use replay. Only speculatable coordinate expressions may cross a possibly
// empty loop; reads still require nonempty execution and source stability.
// Branch boundaries are preserved.
mlir::FailureOr<llvm::SmallVector<ProducerReplayGroup>> groupProducerReplays(
    const ProducerReplay &payload, mlir::Value result, mlir::Operation *producer,
    mlir::Block *allocationScope, llvm::ArrayRef<ProducerReplayUse> uses,
    bool removesProducer, StorageAnalysis &storage);

// Both structured fusion and indexed replay can remove the last read. The
// producer has already passed the shared read-only payload legality query.
void eraseUnusedProducer(mlir::linalg::GenericOp producer);

} // namespace intent::cpu

#endif
