#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace intentdsl::dsa {
using namespace mlir;
namespace {
// Reuse a fetched fragment and its pointwise consumers within a stage.
// A write or unknown operation invalidates the read set, including aliases.
struct ReuseFragments : PassWrapper<ReuseFragments, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseFragments)
  void runOnOperation() final {
    getOperation().walk([](Block *block) {
      DenseMap<llvm::hash_code, SmallVector<Operation *>> available;
      for (auto &op : llvm::make_early_inc_range(*block)) {
        auto name = op.getName().stripDialect();
        if (op.getName().getDialectNamespace() != "block" ||
            !llvm::is_contained(
                ArrayRef<StringRef>{"load", "cast", "mul", "add", "sub", "div",
                                    "rsqrt", "sqrt", "exp", "maximum",
                                    "constant", "reduce", "gt", "ne", "select",
                                    "and", "xor", "shl", "shr"},
                name)) {
          available.clear();
          continue;
        }
        auto hash = OperationEquivalence::computeHash(&op,
            OperationEquivalence::directHashValue, OperationEquivalence::ignoreHashValue,
            OperationEquivalence::IgnoreLocations);
        auto &bucket = available[hash];
        auto match = llvm::find_if(bucket, [&](Operation *other) {
          return OperationEquivalence::isEquivalentTo(other, &op,
              OperationEquivalence::IgnoreLocations);
        });
        if (match == bucket.end()) bucket.push_back(&op);
        else {
          op.replaceAllUsesWith((*match)->getResults());
          op.erase();
        }
      }
    });
  }
};
}
void reuseFragments(OpPassManager &pm) { pm.addPass(std::make_unique<ReuseFragments>()); }
}
