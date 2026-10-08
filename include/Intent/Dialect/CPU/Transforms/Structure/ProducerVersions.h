#ifndef INTENT_CPU_TRANSFORMS_STRUCTURE_PRODUCERVERSIONS_H
#define INTENT_CPU_TRANSFORMS_STRUCTURE_PRODUCERVERSIONS_H

#include "Intent/Dialect/CPU/Analysis/ProducerReplay.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include <optional>

namespace intent::cpu {

struct ProducerVersionUse {
  mlir::Operation *operation;
  unsigned input;
};

// All observations after one complete write and before its next complete write
// or release. Every observation must be an exact, transportable input slot.
// These facts are valid only until the first rewrite of this lexical sequence.
struct ProducerVersion {
  llvm::SmallVector<ProducerVersionUse> uses;
};

// The last possible write before this operation in its current execution
// instance. This does not prove complete coverage or permit moving the write's
// inputs. Crossing an enclosing region requires it to preserve the buffer,
// including later iterations; a conditional or partial clobber is not skipped.
struct CurrentBufferWrite {
  mlir::Operation *operation;
  // Potential reads between this write and the query, excluding the query op.
  // An enclosing scope is conservatively counted as a whole observation.
  bool observed;
};
std::optional<CurrentBufferWrite> findCurrentBufferWrite(mlir::Value buffer,
    mlir::Operation *before, StorageAnalysis &storage);

struct UniformBufferValue {
  CurrentBufferWrite definition;
  mlir::Value value;
};
// Follow complete scalar fills/stores and copies at each copy's original read
// point. Later changes to a copy's input cannot change its stored snapshot.
std::optional<UniformBufferValue> queryUniformBufferValue(mlir::Value buffer,
    mlir::Operation *before, StorageAnalysis &storage);

struct BufferVersionObservations {
  llvm::SmallVector<mlir::Operation *> reads;
  mlir::Operation *end = nullptr;
};

// Observe one completed version until its next complete overwrite or release.
// A sequential For yield also ends a version when its private allocation has
// only this complete defining write and dominated reads in that iteration;
// the full alias closure has no other contents observations or execution owner.
// Readers are actual effect owners, including nested scalar reads. The caller
// proves the defining write's coverage and each reader's transport relation.
mlir::FailureOr<BufferVersionObservations> queryBufferVersionObservations(
    mlir::Value buffer, mlir::Operation *completed, StorageAnalysis &storage);

// All observations of this version have been replaced. Later versions retain
// the allocation; otherwise remove the now unused private allocation as well.
void eraseReplayedProducerVersion(mlir::linalg::GenericOp producer);

bool completelyWritesBuffer(mlir::Operation *operation, mlir::Value buffer);

mlir::FailureOr<ProducerVersion> queryProducerVersion(
    mlir::linalg::GenericOp producer, StorageAnalysis &storage);

// The caller proves that the buffer is fully initialized after this operation.
// Query only that completed version's subsequent observations and termination.
mlir::FailureOr<ProducerVersion> queryCompletedBufferVersion(
    mlir::Value buffer, mlir::Operation *completed, StorageAnalysis &storage);

// Select an existing complete copy as the representative of a stored version.
// Retarget the defining write and its reads; do not replay any computation.
bool forwardProducerVersionCopies(mlir::func::FuncOp function);

// In addition to ordinary replay, permit a source's current element to be read
// immediately before its own injective pointwise overwrite. The source must be
// the exact output descriptor at the exact output coordinates, not an alias.
bool canReplayProducerVersionAt(const ProducerReplay &payload,
    mlir::linalg::GenericOp producer, mlir::Operation *consumer,
    mlir::AffineMap coordinates, StorageAnalysis &storage);

} // namespace intent::cpu

#endif
