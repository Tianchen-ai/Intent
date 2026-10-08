#ifndef INTENT_DIALECT_GPU_ANALYSIS_ITERATIONDEPENDENCIES_H
#define INTENT_DIALECT_GPU_ANALYSIS_ITERATIONDEPENDENCIES_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

namespace intent::gpu {

class ScanOp;

// Conditional independence of a current unit-step, carry-free iteration.
// The caller must preserve each iteration's internal order and materialize
// injective-layout guards for guardedViews and non-overlap guards for every
// disjointViews pair before reordering iterations. These are actual resources,
// not an execution plan; unknown effects or coordinate relations fail the query.
struct IndependentIterationAccesses {
  llvm::SmallVector<mlir::Value> guardedViews;
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> disjointViews;
  // Additional conditions for active count-prefix coordinates: the complete
  // logical member count must not exceed the given inclusive integer limit.
  // The caller must materialize these conditions before reordering iterations.
  llvm::SmallVector<std::pair<mlir::Value, int64_t>> countUpperBounds;
};

mlir::FailureOr<IndependentIterationAccesses>
queryIndependentIterationAccesses(mlir::scf::ForOp loop);

// Also consider an explicitly supplied forward inclusive integer count scan.
// Unknown count/predicate relations retain only the original affine proof domain.
mlir::FailureOr<IndependentIterationAccesses>
queryIndependentIterationAccesses(mlir::scf::ForOp loop, ScanOp countPrefix);

} // namespace intent::gpu

#endif
