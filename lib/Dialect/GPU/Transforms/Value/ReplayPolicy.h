#ifndef INTENT_GPU_TRANSFORMS_VALUE_REPLAYPOLICY_H
#define INTENT_GPU_TRANSFORMS_VALUE_REPLAYPOLICY_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {

// A current, fully realized snapshot covers this whole logical interval. This
// proves the saved lanes only; replay padding still needs its own value proof.
mlir::FailureOr<MakeRangeOp> completeSnapshotRange(
    mlir::Value value, unsigned axis, PhysicalProgramAnalysis &analysis);

// Slice an existing physical snapshot, preserving its own proven tail value.
// An empty result is a normal nonmatch; this never grows source coverage.
mlir::FailureOr<mlir::Value> materializeSnapshotSlice(
    mlir::OpBuilder &builder, mlir::Value value, unsigned axis,
    PhysicalExprAttr extent, mlir::Value coordinates, mlir::Operation *anchor,
    AxisMapAttr resultMapping = {});

// Profitability for one rewrite of the named consumers. Legality and coordinate
// transport continue to belong to the existing replay and projection queries.
class ReplayPolicy {
public:
  ReplayPolicy(mlir::func::FuncOp kernel, mlir::ValueRange roots,
               llvm::ArrayRef<mlir::Operation *> replacedConsumers,
               llvm::function_ref<bool(mlir::Value)> rebuilt = {});

  // Only work in this rewrite's producer slice is considered. Bound values and
  // scalar captures are frontiers; work that dies with the consumers is moved,
  // not duplicated. This is a profitability query, not replay permission.
  bool duplicatesExpensiveWork(mlir::Value root,
                              const mlir::IRMapping *bindings = nullptr) const;

  // The current producer has no surviving use outside this rewrite's slice.
  // Replay legality and actual removal remain obligations of the caller.
  bool removesProducer(mlir::Value value) const;

  bool retains(mlir::Value value, mlir::Operation *anchor,
               const mlir::IRMapping *bindings = nullptr) const;

  // Early traversal selection still owns full coverage for a shared producer
  // whose physical slice is not available yet. Keep that existing route until
  // the local frontier can actually replace it.
  bool preservesSharedTraversal(mlir::Value root, PhysicalSourceAxis source,
                                int64_t dimension) const;

  // An empty successful value means that this existing snapshot cannot be
  // sliced here; the caller keeps its original replay route. This never grows
  // the source's physical coverage or allocates new retained storage.
  mlir::FailureOr<mlir::Value> retainSlice(
      mlir::OpBuilder &builder, mlir::Value value, unsigned axis,
      PhysicalExprAttr extent, mlir::Value coordinates,
      mlir::Operation *anchor, AxisMapAttr resultMapping = {},
      const mlir::IRMapping *bindings = nullptr) const;

  // Bind profitable, representable subexpressions before replaying the cheap
  // suffix. Mappings belong to the caller's single coordinate/scope selection.
  mlir::LogicalResult bindSlices(
      mlir::OpBuilder &builder, mlir::Value root, PhysicalSourceAxis source,
      int64_t dimension, PhysicalExprAttr extent, mlir::Value coordinates,
      mlir::Operation *anchor, mlir::IRMapping &mapping,
      AxisMapAttr resultMapping = {},
      llvm::ArrayRef<int64_t> selectedDimensions = {}) const;

private:
  bool survives(mlir::Operation *operation,
                llvm::DenseSet<mlir::Operation *> &active) const;

  mlir::func::FuncOp kernel;
  llvm::DenseSet<mlir::Operation *> slice;
  llvm::SmallVector<mlir::Operation *> consumers;
};

} // namespace intent::gpu

#endif
