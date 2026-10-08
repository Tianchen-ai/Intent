#ifndef INTENT_DIALECT_CPU_ANALYSIS_STORAGE_H
#define INTENT_DIALECT_CPU_ANALYSIS_STORAGE_H

#include "Intent/Analysis/BufferStorage.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"

namespace intent::cpu {

// Check that explicit release operands connect to owned storage, no heap origin
// is disconnected from all releases, and unconditional lexical frees precede
// no later accesses. Native dealloc verifies its condition/retention schema;
// its transformation establishes the complex control-flow ownership algorithm.
// This structural audit is not a second path-sensitive deallocation proof.
mlir::LogicalResult verifyStorageOwnership(mlir::func::FuncOp function);

using StorageOriginFacts = intent::BufferStorageOriginFacts;
using StorageEffect = intent::BufferStorageEffect;
using StorageEffects = intent::BufferStorageEffects;
using StorageAliasFacts = intent::BufferStorageAliasFacts;
using StorageLifetime = intent::BufferStorageLifetime;

// A closed upper bound for a current descriptor dimension. This query does not
// use observed runtime shapes or interpret wrapping index arithmetic as affine.
std::optional<int64_t> constantDimensionUpperBound(mlir::Value memory, unsigned axis);

inline bool isStorageAliasOperation(mlir::Operation *operation) {
  return isBufferStorageAliasOperation(operation);
}

// CPU contracts supplement the shared current-IR buffer analysis. Task/formal
// identity and public ABI permissions remain CPU facts, not generic aliases.
class StorageAnalysis : public intent::BufferStorageAnalysis {
public:
  explicit StorageAnalysis(mlir::func::FuncOp function);
  intent::ViewType externalView(mlir::Value memory) const;
  bool isReadOnly(mlir::Value memory) const;

private:
  mlir::func::FuncOp function;
};

} // namespace intent::cpu
#endif
