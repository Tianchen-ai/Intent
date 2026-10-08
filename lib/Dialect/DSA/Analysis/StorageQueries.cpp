#include "Intent/Dialect/DSA/Analysis/Storage.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

using namespace mlir;

namespace intent::dsa {

FailureOr<SmallVector<Operation *>> StorageAnalysis::writers(Value memory) {
  auto summary = accesses(memory);
  if (!summary.complete)
    return failure();
  SmallVector<Operation *> result;
  for (const auto &entry : summary.entries) {
    if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(entry.effect.getEffect()))
      continue;
    if (!isa<MemoryEffects::Write>(entry.effect.getEffect()))
      return failure();
    if (!llvm::is_contained(result, entry.operation))
      result.push_back(entry.operation);
  }
  return result;
}

Operation *StorageAnalysis::uniqueWriter(Value memory) {
  auto writing = writers(memory);
  return succeeded(writing) && writing->size() == 1 ? writing->front() : nullptr;
}

bool StorageAnalysis::allUsesCompleteBefore(Value memory, Operation *point,
                                           Operation *except) {
  auto summary = accesses(memory);
  if (!summary.complete)
    return false;
  auto source = origins(memory);
  for (Operation *scope = point->getParentOp(); scope; scope = scope->getParentOp()) {
    if (!isa<LoopLikeOpInterface>(scope))
      continue;
    // Static dominance does not prove a last dynamic use across a backedge.
    // Only storage instantiated inside this iteration can be consumed here.
    if (llvm::any_of(source.values, [&](Value origin) {
          Operation *definition = origin.getDefiningOp();
          return !definition || !scope->isProperAncestor(definition);
        }))
      return false;
  }
  DominanceInfo dominance(point->getParentOfType<func::FuncOp>());
  for (const auto &entry : summary.entries) {
    if (entry.operation == except ||
        isa<MemoryEffects::Allocate>(entry.effect.getEffect()))
      continue;
    auto completion = completionOfUse(entry.operation);
    if (failed(completion) ||
        !dominance.properlyDominates(*completion, point))
      return false;
  }
  return true;
}

Operation *StorageAnalysis::lastWriterBefore(Value memory, Operation *read) {
  Value origin = uniqueOrigin(memory);
  if (!origin || !origin.getDefiningOp<memref::AllocaOp>() ||
      !aliases(origin).complete)
    return nullptr;
  DominanceInfo dominance(read->getParentOfType<func::FuncOp>());
  Operation *cursor = read;
  while (true) {
    for (Operation *previous = cursor->getPrevNode(); previous;
         previous = previous->getPrevNode()) {
      auto summary = effects(previous);
      if (!summary.complete)
        return nullptr;
      bool written = false;
      for (const auto &entry : summary.entries) {
        if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(entry.effect.getEffect()) ||
            disjoint(memory, entry.effect.getValue()))
          continue;
        // A conditional/loop write is a clobber, not one unconditional writer.
        if (entry.operation != previous ||
            !isa<MemoryEffects::Write>(entry.effect.getEffect()))
          return nullptr;
        written = true;
      }
      if (!written)
        continue;
      auto completion = completionOfUse(previous);
      if (failed(completion) || !dominance.properlyDominates(*completion, read))
        return nullptr;
      return previous;
    }
    Operation *parent = cursor->getParentOp();
    if (!parent || !dominance.dominates(memory, parent))
      return nullptr;
    // An outer definition can reach any loop iteration only when backedges
    // preserve it. Ordinary conditional entry cannot revisit a prior branch.
    if (isa<LoopLikeOpInterface>(parent)) {
      if (!preservesContents(parent, memory))
        return nullptr;
    } else if (!isa<scf::IfOp>(parent)) {
      return nullptr;
    }
    cursor = parent;
  }
}

} // namespace intent::dsa
