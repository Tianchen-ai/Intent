#ifndef INTENT_DIALECT_GPU_TRANSFORMS_STORAGE_FRAGMENTSNAPSHOT_H
#define INTENT_DIALECT_GPU_TRANSFORMS_STORAGE_FRAGMENTSNAPSHOT_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu {

// Return the predicate of a common enclosing If arm for every selected reader.
// The If must share the source definition's block and follow that definition;
// only an else-arm predicate needs new scalar IR immediately before the If.
// No predicate is returned when this control relation is unknown.
mlir::Value materializeFragmentSnapshotReadDomain(
    mlir::Value source, llvm::ArrayRef<GatherOp> readers);

// Save the completed physical SSA lanes at their definition (or at the current
// block argument's entry). The fresh program-private buffer has position axes;
// this neither replays the producer nor chooses whether storage is profitable.
// An optional scalar i1 predicates the complete initialization. A later same-
// block definition may delay this private store; source evaluation never moves.
// The caller must place every snapshot read under enabled's true control path.
mlir::FailureOr<mlir::Value> materializeFragmentSnapshot(
    mlir::Value source, mlir::Value enabled = {});

// Read a previously materialized snapshot at the original gather point. Preserve
// its coordinate/result relation, validity, fill and logical index dtype. The
// caller replaces/erases the gather after updating its own dependency mappings.
mlir::FailureOr<mlir::Value> loadFragmentSnapshot(GatherOp gather,
                                                mlir::Value snapshot);

} // namespace intent::gpu

#endif
