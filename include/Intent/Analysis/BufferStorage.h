#ifndef INTENT_ANALYSIS_BUFFERSTORAGE_H
#define INTENT_ANALYSIS_BUFFERSTORAGE_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <optional>

namespace intent {

void registerBufferStorageInterfaces(mlir::DialectRegistry &registry);

struct BufferStorageOriginFacts {
  llvm::SmallVector<mlir::Value> values;
  // A known origin may be a borrowed invocation formal. Completeness describes
  // identity, not freshness or disjointness from other origins.
  bool complete = false;
  mlir::Value uniqueOrigin() const;
};

struct BufferStorageEffect {
  mlir::Operation *operation;
  mlir::MemoryEffects::EffectInstance effect;
};

struct BufferStorageEffects {
  // Only addressable storage effects. Other explicitly classified resources
  // retain their native MLIR effects without becoming writes to every buffer.
  llvm::SmallVector<BufferStorageEffect> entries;
  bool complete = true;
  bool ordered = false;
};

struct BufferStorageAliasFacts {
  mlir::Value root;
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallVector<mlir::Operation *> users;
  bool complete = true;
};

bool isBufferStorageAliasOperation(mlir::Operation *operation);

struct BufferStorageLifetime {
  mlir::memref::AllocOp allocation;
  mlir::memref::DeallocOp end;
  BufferStorageAliasFacts aliases;
  bool contains(mlir::Operation *operation) const;
};

// Execution-family facts supplement native MLIR flow and effects. These hooks
// query current IR; they do not replace alias edges or store an execution plan.
struct BufferStoragePolicy {
  std::function<bool(mlir::BlockArgument)> isBorrowedArgument;
  std::function<bool(mlir::Value, mlir::Value)> provenDisjointOrigins;
  std::function<bool(mlir::Operation *)> isOrderingBarrier;
  std::function<bool(mlir::SideEffects::Resource *)> isNonStorageResource;
};

// One current-IR snapshot. Recreate after changing operands, control flow,
// views, storage, or effects. Async completion ranges remain family facts.
class BufferStorageAnalysis {
public:
  explicit BufferStorageAnalysis(mlir::func::FuncOp function,
                                 BufferStoragePolicy policy = {});

  BufferStorageOriginFacts origins(mlir::Value memory) const;
  mlir::Value uniqueOrigin(mlir::Value memory) const;
  // Forward flow from the supplied value. Pass an allocation origin when all
  // of its derived descriptors and uses are required.
  BufferStorageAliasFacts aliases(mlir::Value root) const;
  // A single lexical release must cover every alias use. This is not a second
  // path-sensitive ownership algorithm or an asynchronous completion proof.
  std::optional<BufferStorageLifetime>
  lifetime(mlir::memref::AllocOp allocation) const;
  BufferStorageEffects effects(mlir::Operation *scope) const;
  // All current accesses that may affect this storage, with each operation's
  // effects included once. This is a writer/read set, not a motion proof;
  // unchangedBetween/readStable limit their checks to an execution interval.
  BufferStorageEffects accesses(mlir::Value memory);
  bool disjoint(mlir::Value first, mlir::Value second);
  // Coexisting instances during this scope. Unlike static disjoint(), this may
  // distinguish a new heap instance from an incoming loop-carried descriptor;
  // it must not filter a whole-lifetime alias/access closure.
  bool disjointAt(mlir::Value first, mlir::Value second, mlir::Operation *scope);
  // A content observation may cross ordering alone; moving a read may not.
  bool preservesContents(mlir::Operation *scope, mlir::Value memory);
  bool preserves(mlir::Operation *scope, mlir::Value memory);

  // Observe after 'from' and before 'to'; neither endpoint participates. Both
  // endpoints belong to the same lexical sequence.
  bool unchangedBetween(mlir::Value memory, mlir::Operation *from,
                        mlir::Operation *to);
  bool contentsUnchangedBetween(mlir::Value memory, mlir::Operation *from,
                                mlir::Operation *to);
  // Replay a read in 'from' at 'to', including the producer and enclosing
  // consumer's effects and nested execution regions.
  bool readStable(mlir::Value memory, mlir::Operation *from,
                  mlir::Operation *to);

private:
  const mlir::BufferViewFlowAnalysis &getFlow() const;
  mlir::BufferOriginAnalysis &getBufferOrigins();
  bool unchangedBetween(mlir::Value memory, mlir::Operation *from,
                        mlir::Operation *to, bool respectOrdering);
  bool preservesContents(const BufferStorageEffects &effects, mlir::Value memory,
                         mlir::Operation *scope);
  bool isLiveAt(mlir::Value memory, mlir::Operation *operation) const;
  mlir::func::FuncOp function;
  BufferStoragePolicy policy;
  // Native whole-function graphs are built only by queries that need them.
  // Both belong to this snapshot and must be discarded after an IR mutation.
  mutable std::optional<mlir::BufferViewFlowAnalysis> flow;
  std::optional<mlir::BufferOriginAnalysis> bufferOrigins;
  mlir::AliasAnalysis aliasAnalysis;
  mlir::DominanceInfo dominance;
};

} // namespace intent

#endif
