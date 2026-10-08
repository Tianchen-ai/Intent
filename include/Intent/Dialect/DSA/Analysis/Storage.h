#ifndef INTENT_DIALECT_DSA_ANALYSIS_STORAGE_H
#define INTENT_DIALECT_DSA_ANALYSIS_STORAGE_H

#include "Intent/Analysis/BufferStorage.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"

namespace intent::dsa {

// Standard memref flow/effects with DSA entry, borrowed-helper and completion
// facts. Recreate after rewriting the program; this does not own storage plans.
class StorageAnalysis : public intent::BufferStorageAnalysis {
public:
  explicit StorageAnalysis(mlir::func::FuncOp function);

  // Synchronous uses complete at the operation. An asynchronous use needs a
  // reachable, explicit completion in the same iteration; failure is unknown.
  mlir::FailureOr<mlir::Operation *> completionOfUse(mlir::Operation *use) const;

  mlir::FailureOr<llvm::SmallVector<mlir::Operation *>> writers(mlir::Value memory);
  mlir::Operation *uniqueWriter(mlir::Value memory);
  mlir::Operation *lastWriterBefore(mlir::Value memory, mlir::Operation *read);
  bool allUsesCompleteBefore(mlir::Value memory, mlir::Operation *point,
                            mlir::Operation *except = nullptr);
};

struct StorageLifetime {
  mlir::memref::AllocaOp allocation;
  uint64_t start, finish;
};
llvm::SmallVector<StorageLifetime>
analyzeStorageLifetimes(mlir::func::FuncOp function);

} // namespace intent::dsa
#endif
