#include "compilerinfra/Proofs.h"
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
// Reuse a fetched fragment and its pure consumers within a single block using
// CompilerInfra's conservative MemoryEffectOpInterface + proveValueReplay model:
// - Any operation without MemoryEffectOpInterface is treated as unknown,
//   never reused, and invalidates all cached entries;
// - Any operation with Write/Free invalidates cached read-bearing entries that
//   may alias the modified resource, and non-Read operations are never cached;
// - Read-bearing operations require proveValueReplay(other, &op, {}, &aa).proven.
struct ReuseFragments : PassWrapper<ReuseFragments, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseFragments)
  void runOnOperation() final {
    ModuleOp mod = getOperation();
    AliasAnalysis aa(mod);
    aa.addAnalysisImplementation(LocalAliasAnalysis());

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
  }
};
} // namespace

void reuseFragments(OpPassManager &pm) {
  pm.addPass(std::make_unique<ReuseFragments>());
}
} // namespace intentdsl::dsa
