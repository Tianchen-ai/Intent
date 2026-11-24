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
bool isPureBlockExpr(Operation &op) {
  if (op.getName().getDialectNamespace() != "block")
    return false;
  return llvm::is_contained(
      ArrayRef<StringRef>{"cast", "mul", "add", "sub", "div", "rsqrt", "sqrt",
                          "exp", "maximum", "constant", "reduce", "gt", "ne",
                          "select", "and", "xor", "shl", "shr"},
      op.getName().stripDialect());
}

// Reuse a fetched fragment and its pointwise consumers within a stage.
// Standard MemoryEffectOpInterface regions use CompilerInfra's conservative
// proveValueReplay; unmodeled block-stage ops invalidate on any write/unknown.
struct ReuseFragments : PassWrapper<ReuseFragments, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseFragments)
  void runOnOperation() final {
    ModuleOp mod = getOperation();
    AliasAnalysis aa(mod);
    aa.addAnalysisImplementation(LocalAliasAnalysis());

    mod.walk([&](Block *block) {
      DenseMap<llvm::hash_code, SmallVector<Operation *>> available;
      for (auto &op : llvm::make_early_inc_range(*block)) {
        if (op.getNumRegions() > 0 || op.hasTrait<OpTrait::IsTerminator>()) {
          available.clear();
          continue;
        }
        bool isBlockLoad = op.getName().getStringRef() == "block.load";
        bool isPureBlk = isPureBlockExpr(op);
        auto memIface = dyn_cast<MemoryEffectOpInterface>(&op);

        if (!isBlockLoad && !isPureBlk && !memIface) {
          available.clear();
          continue;
        }
        if (memIface && !isBlockLoad && !isPureBlk) {
          SmallVector<MemoryEffects::EffectInstance, 4> effs;
          memIface.getEffects(effs);
          bool hasWriteOrAlloc = llvm::any_of(effs, [](const auto &e) {
            return !isa<MemoryEffects::Read>(e.getEffect());
          });
          if (hasWriteOrAlloc)
            continue;
        }

        auto hash = OperationEquivalence::computeHash(
            &op, OperationEquivalence::directHashValue,
            OperationEquivalence::ignoreHashValue,
            OperationEquivalence::IgnoreLocations);
        auto &bucket = available[hash];
        auto match = llvm::find_if(bucket, [&](Operation *other) {
          if (!OperationEquivalence::isEquivalentTo(
                  other, &op, OperationEquivalence::IgnoreLocations))
            return false;
          if (memIface) {
            return compilerinfra::proveValueReplay(other, &op, {}, &aa).proven;
          }
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
