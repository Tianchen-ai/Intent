#include "compilerinfra/Proofs.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/block/ir/OwnerReductions.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace intentdsl::dsa {
using namespace mlir;
namespace {
// Each projected view names its public storage slot. Only the bound call's
// disjoint groups prove separation; distinct arguments alone do not.
struct BoundAliases {
  DenseMap<Value, int64_t> groups;
  AliasResult alias(Value a, Value b) {
    auto x = groups.find(a), y = groups.find(b);
    return x != groups.end() && y != groups.end() && x->second != y->second
               ? AliasResult::NoAlias : AliasResult::MayAlias;
  }

  ModRefResult getModRef(Operation *op, Value loc) {
    auto mem = dyn_cast<MemoryEffectOpInterface>(op);
    if (!mem)
      return ModRefResult::getModAndRef();
    SmallVector<MemoryEffects::EffectInstance> effects;
    mem.getEffects(effects);
    auto result = ModRefResult::getNoModRef();
    for (const auto &effect : effects) {
      if (effect.getValue() && alias(effect.getValue(), loc).isNo())
        continue;
      if (isa<MemoryEffects::Read>(effect.getEffect()))
        result = result.merge(ModRefResult::getRef());
      else
        result = result.merge(ModRefResult::getMod());
    }
    return result;
  }
};
// Reuse a fetched fragment and its pure consumers within a single block using
// CompilerInfra's conservative MemoryEffectOpInterface + proveValueReplay model:
// - Any operation without MemoryEffectOpInterface is treated as unknown,
//   never reused, and invalidates all cached entries;
// - Any operation with Write/Free invalidates cached read-bearing entries that
//   may alias the modified resource, and non-Read operations are never cached;
// - Read-bearing operations require proveValueReplay(other, &op, {}, &aa).proven.
struct ReuseFragments : PassWrapper<ReuseFragments, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseFragments)
  compilerinfra::TargetProfile profile;
  explicit ReuseFragments(const compilerinfra::TargetProfile &p) : profile(p) {}
  void runOnOperation() final {
    ModuleOp mod = getOperation();
    AliasAnalysis aa(mod);
    aa.addAnalysisImplementation(LocalAliasAnalysis());
    if (auto *facts = profile.services.get<CallFacts>()) {
      BoundAliases aliases;
      for (auto fn : mod.getOps<func::FuncOp>()) {
        for (auto arg : fn.getArguments()) {
          if (!isa<MemRefType>(arg.getType()))
            continue;
          auto source = fn.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), "kir.source");
          int64_t slot = source ? source.getInt() : arg.getArgNumber();
          if (slot < 0 || size_t(slot) >= facts->arguments.size())
            continue;
          if (auto *record = facts->arguments[slot].getAsObject())
            if (auto group = record->getInteger("alias_group"); group && *group >= 0)
              aliases.groups.try_emplace(arg, *group);
        }
      }
      aa.addAnalysisImplementation(std::move(aliases));
    }

    mod.walk([&](Block *block) {
      DenseMap<llvm::hash_code, SmallVector<Operation *>> available;

      auto evictAliasingReads = [&](Value writeTarget) {
        for (auto &kv : available) {
          llvm::erase_if(kv.second, [&](Operation *cached) {
            auto cachedMem = dyn_cast<MemoryEffectOpInterface>(cached);
            if (!cachedMem)
              return true;
            SmallVector<MemoryEffects::EffectInstance, 4> ceffs;
            cachedMem.getEffects(ceffs);
            for (const auto &ce : ceffs) {
              if (!isa<MemoryEffects::Read>(ce.getEffect()))
                return true;
              if (!writeTarget || !ce.getValue() ||
                  !aa.alias(ce.getValue(), writeTarget).isNo())
                return true;
            }
            return false;
          });
        }
      };

      for (auto &op : llvm::make_early_inc_range(*block)) {
        if (op.getNumRegions() > 0 || op.hasTrait<OpTrait::IsTerminator>()) {
          available.clear();
          continue;
        }
        auto memIface = dyn_cast<MemoryEffectOpInterface>(&op);
        if (!memIface) {
          available.clear();
          continue;
        }

        SmallVector<MemoryEffects::EffectInstance, 4> effs;
        memIface.getEffects(effs);
        bool hasNonRead = false;
        for (const auto &e : effs) {
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(e.getEffect())) {
            evictAliasingReads(e.getValue());
            hasNonRead = true;
          } else if (!isa<MemoryEffects::Read>(e.getEffect())) {
            hasNonRead = true;
          }
        }
        if (hasNonRead)
          continue;

        bool hasRead = !effs.empty();
        auto hash = OperationEquivalence::computeHash(
            &op, OperationEquivalence::directHashValue,
            OperationEquivalence::ignoreHashValue,
            OperationEquivalence::IgnoreLocations);
        auto &bucket = available[hash];
        auto match = llvm::find_if(bucket, [&](Operation *other) {
          if (!OperationEquivalence::isEquivalentTo(
                  other, &op, OperationEquivalence::IgnoreLocations))
            return false;
          if (hasRead)
            return compilerinfra::proveValueReplay(other, &op, {}, &aa).proven;
          return true;
        });
        if (match == bucket.end()) {
          bucket.push_back(&op);
        } else {
          op.replaceAllUsesWith((*match)->getResults());
          op.erase();
        }
      }
    });
    for (auto fn : mod.getOps<func::FuncOp>())
      block::fuseOwnerReductions(fn, profile, /*partitionOutputs=*/false);
  }
};
} // namespace

void reuseFragments(OpPassManager &pm, const compilerinfra::TargetProfile &p) {
  pm.addPass(std::make_unique<ReuseFragments>(p));
}
} // namespace intentdsl::dsa
