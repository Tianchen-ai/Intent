#ifndef INTENT_DIALECT_GPU_ANALYSIS_RESOURCEALIAS_H
#define INTENT_DIALECT_GPU_ANALYSIS_RESOURCEALIAS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace intent::gpu {

// Allocation identity in the current program. A resource forwarded through
// control flow remains an alias of every possible incoming allocation.
// Discard this query cache after changing resource definitions or control flow.
class ResourceAliasAnalysis {
public:
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);

  // In addition to global allocation facts, consume a launch-time view-overlap
  // predicate proved false by an actual enclosing If arm. This is a byte-span
  // fact for the exact external views, not a new global allocation identity;
  // no branch fact survives its join or is cached as an AliasResult.
  bool disjointAt(mlir::Value lhs, mlir::Value rhs, mlir::Operation *context);

private:
  struct Roots {
    llvm::SmallVector<mlir::Value> allocations;
    bool complete = true;
  };
  const Roots &roots(mlir::Value value);
  llvm::DenseMap<mlir::Value, Roots> cache;
};

} // namespace intent::gpu

#endif
