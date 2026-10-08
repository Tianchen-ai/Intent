#ifndef INTENT_DIALECT_GPU_ANALYSIS_REDUCTIONCONSUMERS_H
#define INTENT_DIALECT_GPU_ANALYSIS_REDUCTIONCONSUMERS_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

namespace intent::gpu {

struct ReductionConsumerProjection {
  mlir::Value value;
  PhysicalAxisProjection axis;
};

// Current-IR facts for one summary and its same-block pointwise output group.
// Operations and values remain at their original anchors. Captures are existing
// SSA values available before the reduction, or its summary results. Pure
// summary-dependent operations and their pure operands after the reduction
// belong to outputOperations, even without the selected member axis. Captures
// grant no replay permission for their definitions.
struct ReductionConsumerGroup {
  ReduceOp reduce;
  unsigned reductionAxis = 0;
  MakeRangeOp range;
  llvm::SmallVector<StoreOp> stores;
  llvm::SmallVector<mlir::Operation *> producerOperations;
  llvm::SmallVector<mlir::Operation *> outputOperations;
  llvm::SmallVector<mlir::Value> wholeProducers;
  llvm::SmallVector<ReductionConsumerProjection> projections;
  llvm::SmallVector<LoadOp> sourceLoads;
  llvm::SmallVector<mlir::Value> captures;
};

// Discover a closed whole-value group after ordinary reduction normalization
// and tuple fusion. This does not choose a schedule or prove memory stability:
// the rewriter must check actual read anchors, aliases and complete loop effects
// in its chosen control domain, and requery facts after every mutation.
mlir::FailureOr<ReductionConsumerGroup>
queryReductionConsumerGroup(ReduceOp reduce, unsigned reductionAxis);

} // namespace intent::gpu

#endif
