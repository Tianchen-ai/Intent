#ifndef INTENT_DIALECT_CPU_ANALYSIS_UNIFORMVALUES_H
#define INTENT_DIALECT_CPU_ANALYSIS_UNIFORMVALUES_H

#include "Intent/Analysis/UniformValues.h"
#include "Intent/Analysis/BufferStorage.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <utility>

namespace intent::cpu {

class StorageAnalysis;

// Whole-view aliases have the same uniform contents. Partial slices retain
// their own key; uniform facts about their complete allocation may still apply.
mlir::Value canonicalUniformMemory(mlir::Value value);

mlir::Attribute foldUniformComputation(mlir::linalg::GenericOp operation,
    const UniformBindings &operands, const UniformBindings &scalarFacts = UniformBindings(),
    unsigned outputIndex = 0);

struct UniformComputationFacts {
  UniformBindings operands;
  llvm::SmallVector<mlir::Attribute> outputs;
  bool canSubstituteInputs = false;
};

// Sequential facts about current buffer contents, using immutable scalar SSA
// bindings and the caller's current storage-analysis snapshot. This object does
// not rewrite IR or choose a representation for known uniform values.
class UniformMemoryAnalysis {
public:
  explicit UniformMemoryAnalysis(StorageAnalysis &storage,
                                 UniformBindings scalarFacts = {});
  void setStorageAnalysis(StorageAnalysis &current) { storage = &current; }
  void setFacts(UniformBindings facts) { memory = std::move(facts); }
  const UniformBindings &getFacts() const { return memory; }

  mlir::Attribute read(mlir::Value value) const;
  void write(mlir::Value value, mlir::Attribute constant);
  void invalidate(mlir::Operation *operation);
  UniformComputationFacts visit(mlir::linalg::GenericOp operation);
  void visit(mlir::Operation *operation);

private:
  void invalidate(const intent::BufferStorageEffects &effects);
  UniformComputationFacts evaluate(mlir::linalg::GenericOp operation) const;

  StorageAnalysis *storage;
  UniformBindings scalarFacts;
  UniformBindings memory;
  UniformValueAnalysis values;
};

}
#endif
