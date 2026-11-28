#pragma once
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

namespace intentdsl::kir {
// Value-number read-only row expressions within one memory epoch. Replace
// only equal reductions: intervening stores or opaque effects close the epoch.
// This preserves the type, per-op permissions and order of every product.
inline void sharePartialSums(mlir::Block &body) {
  using namespace mlir;
  auto isFold = [](Operation &op) {
    auto name = op.getName().getStringRef();
    return name == "kir.sum" || name == "kir.max_num" || name == "kir.welford";
  };
  if (llvm::count_if(body, isFold) < 2)
    return;
  IRMapping leaders;
  DenseMap<llvm::hash_code, SmallVector<Operation *, 1>> expressions;
  for (Operation &op : llvm::make_early_inc_range(body)) {
    bool fold = isFold(op);
    auto name = op.getName().getStringRef();
    bool expression =
        llvm::StringSwitch<bool>(name)
            .Cases("kir.read", "kir.read_owner", "kir.constant", "kir.cast",
                   true)
            .Cases("kir.add", "kir.sub", "kir.mul", "kir.div", "kir.maximum",
                   true)
            .Cases("kir.rsqrt", "kir.exp", "kir.sqrt", "kir.field", true)
            .Default(false);
    if (!fold && !expression) {
      expressions.clear();
      leaders.clear();
      continue;
    }
    auto key = OperationEquivalence::computeHash(
        &op, [&](Value v) { return hash_value(leaders.lookupOrDefault(v)); },
        OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
    auto &bucket = expressions[key];
    auto found = llvm::find_if(bucket, [&](Operation *prev) {
      return OperationEquivalence::isEquivalentTo(
          prev, &op,
          [&](Value a, Value b) {
            return success(leaders.lookupOrDefault(a) ==
                           leaders.lookupOrDefault(b));
          },
          nullptr, OperationEquivalence::IgnoreLocations);
    });
    if (found == bucket.end()) {
      bucket.push_back(&op);
      continue;
    }
    Value prev = (*found)->getResult(0);
    if (fold) {
      op.getResult(0).replaceAllUsesWith(prev);
      op.erase();
    } else {
      leaders.map(op.getResult(0), prev);
    }
  }
}
} // namespace intentdsl::kir
