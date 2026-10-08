#include "MemoryAccess.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include <utility>

using namespace mlir;

namespace intent::cpu {
namespace {

class MemoryValues {
public:
  explicit MemoryValues(func::FuncOp function)
      : storage(function), dominance(function) {}

  void analyze(Block &block, SmallVector<detail::MemoryAccess> available = {}) {
    SmallVector<detail::MemoryAccess> pendingStores;
    for (Operation &operation : block) {
      if (auto access = detail::memoryAccess(&operation)) {
        if (!access->write) {
          auto found = llvm::find_if(llvm::reverse(available), [&](const auto &previous) {
            return detail::sameMemoryAccess(previous, *access, replacements) &&
                   dominance.properlyDominates(resolve(previous.value), &operation);
          });
          if (found != llvm::reverse(available).end()) {
            replacements.map(access->value, resolve(found->value));
            replacedLoads.push_back(access->operation);
            continue;
          }
          forgetAliasing(pendingStores, access->memory);
          available.push_back(std::move(*access));
        } else {
          // These stores are in one lexical block. The later complete store
          // executes after the earlier one, and every intervening read has
          // already removed the earlier store from the candidate set.
          for (const auto &previous : pendingStores)
            if (detail::sameMemoryAccess(previous, *access, replacements))
              deadStores.insert(previous.operation);
          llvm::erase_if(pendingStores, [&](const auto &previous) {
            return deadStores.contains(previous.operation);
          });
          forgetAliasing(available, access->memory);
          pendingStores.push_back(*access);
          available.push_back(std::move(*access));
        }
        continue;
      }

      auto effects = storage.effects(&operation);
      for (Region &region : operation.getRegions()) {
        for (Block &nested : region) {
          SmallVector<detail::MemoryAccess> inherited;
          if (effects.complete && !effects.ordered &&
              !operation.hasTrait<OpTrait::IsIsolatedFromAbove>())
            for (const auto &access : available)
              if (isa<scf::IfOp>(operation) || storage.preserves(&operation, access.memory))
                inherited.push_back(access);
          // Branch-local values and writes are never published to the parent.
          // Loops inherit only observations unchanged by all iterations.
          // An opaque enclosing effect blocks propagation, not optimization
          // within each body's own definitely executed lexical sequence.
          analyze(nested, std::move(inherited));
        }
      }
      if (!effects.complete || effects.ordered) {
        available.clear();
        pendingStores.clear();
        continue;
      }
      for (const StorageEffect &entry : effects.entries) {
        auto *effect = entry.effect.getEffect();
        Value memory = entry.effect.getValue();
        if (isa<MemoryEffects::Read, MemoryEffects::Free>(effect))
          forgetAliasing(pendingStores, memory);
        if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect))
          forgetAliasing(available, memory);
      }
    }
  }

  bool apply() {
    // No IR changes while StorageAnalysis and DominanceInfo are queried.
    // Resolve the complete replacement chain before erasing any defining op.
    SmallVector<std::pair<Operation *, Value>> replacementsToApply;
    for (Operation *load : replacedLoads)
      replacementsToApply.emplace_back(load, resolve(load->getResult(0)));
    for (auto [load, value] : replacementsToApply) {
      load->getResult(0).replaceAllUsesWith(value);
      load->erase();
    }
    for (Operation *store : deadStores) store->erase();
    return !replacedLoads.empty() || !deadStores.empty();
  }

private:
  Value resolve(Value value) const {
    while (Value replacement = replacements.lookupOrNull(value)) {
      if (replacement == value) break;
      value = replacement;
    }
    return value;
  }

  void forgetAliasing(SmallVectorImpl<detail::MemoryAccess> &accesses, Value memory) {
    llvm::erase_if(accesses, [&](const auto &previous) {
      return !memory || !storage.disjoint(memory, previous.memory);
    });
  }

  StorageAnalysis storage;
  DominanceInfo dominance;
  IRMapping replacements;
  SmallVector<Operation *> replacedLoads;
  llvm::SetVector<Operation *> deadStores;
};

} // namespace

bool reuseMemoryValues(func::FuncOp function) {
  if (function.isExternal()) return false;
  MemoryValues values(function);
  for (Block &block : function.getBody()) values.analyze(block);
  return values.apply();
}

} // namespace intent::cpu
