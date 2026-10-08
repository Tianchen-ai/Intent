#ifndef INTENT_DIALECT_DSA_ANALYSIS_UNIFORMVALUES_H
#define INTENT_DIALECT_DSA_ANALYSIS_UNIFORMVALUES_H

#include "Intent/Dialect/DSA/Analysis/Storage.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"

namespace intent::dsa {

// A current local storage snapshot, not a property of an allocation's lifetime.
// Returned scalar values dominate the actual reader and retain the storage
// element type. Transfers are followed at their original read points. Recreate
// this analysis and its StorageAnalysis after changing IR or memory effects.
class UniformMemoryAnalysis {
public:
  UniformMemoryAnalysis(mlir::func::FuncOp function, StorageAnalysis &storage);
  mlir::Value read(mlir::Value memory, mlir::Operation *reader);

private:
  mlir::Value readSnapshot(mlir::Value memory, mlir::Operation *reader);
  bool completeCounts(mlir::Value rows, mlir::Value columns,
                      mlir::MemRefType shape);
  mlir::func::FuncOp function;
  StorageAnalysis &storage;
  mlir::DominanceInfo dominance;
  llvm::DenseSet<std::pair<mlir::Value, mlir::Operation *>> visiting;
};

} // namespace intent::dsa
#endif
