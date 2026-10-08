#ifndef INTENT_CPU_TRANSFORMS_CONTROL_TRAVERSALFUSION_H
#define INTENT_CPU_TRANSFORMS_CONTROL_TRAVERSALFUSION_H

#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"

namespace intent::cpu::detail {

struct Traversal {
  mlir::Operation *operation;
  mlir::Block *body;
  llvm::SmallVector<mlir::Value> lower, upper, step, coordinates;
};

struct TraversalAccess {
  mlir::Operation *operation;
  mlir::Value memory;
  llvm::SmallVector<mlir::Value> indices;
  bool indexed;
  bool write;
  mlir::AffineMap indexingMap;
};

bool sameTraversalAddress(const TraversalAccess &first,
                          const TraversalAccess &second,
                          const mlir::IRMapping &mapping);
bool sameNestedReadFootprint(const TraversalAccess &first,
                             const TraversalAccess &second,
                             const Traversal &firstTraversal,
                             const Traversal &secondTraversal,
                             const mlir::IRMapping &mapping);
mlir::FailureOr<llvm::SmallVector<TraversalAccess>>
traversalAccesses(const Traversal &traversal, StorageAnalysis &storage);
bool separatesIterations(const TraversalAccess &access,
                         const Traversal &traversal,
                         mlir::DominanceInfo &dominance);
bool containedSubview(mlir::memref::SubViewOp view);

} // namespace intent::cpu::detail
#endif
