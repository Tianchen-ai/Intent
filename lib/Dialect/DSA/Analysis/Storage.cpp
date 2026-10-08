#include "Intent/Dialect/DSA/Analysis/Storage.h"
#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <functional>
#include <limits>

using namespace mlir;

namespace intent::dsa {

StorageAnalysis::StorageAnalysis(func::FuncOp function)
    : BufferStorageAnalysis(function, storagePolicy()) {}

FailureOr<Operation *> StorageAnalysis::completionOfUse(Operation *use) const {
  CompletionScope scope = requiredCompletion(use);
  if (scope == CompletionScope::Immediate)
    return use;
  Operation *cursor = use;
  while (cursor) {
    for (Operation *next = cursor->getNextNode(); next;
         next = next->getNextNode())
      if (completesTransfers(next, scope))
        return next;
    // A wait after an enclosing conditional executes on every path that issued
    // the transfer. Never cross a loop backedge to find a later iteration's wait.
    auto branch = dyn_cast_or_null<scf::IfOp>(cursor->getParentOp());
    if (!branch)
      return failure();
    cursor = branch;
  }
  return failure();
}

SmallVector<StorageLifetime> analyzeStorageLifetimes(func::FuncOp function) {
  DenseMap<Operation *, uint64_t> begin, end;
  uint64_t clock = 0;
  std::function<void(Operation *)> number = [&](Operation *op) {
    begin[op] = clock++;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          number(&nested);
    end[op] = clock++;
  };
  number(function);
  StorageAnalysis storage(function);
  SmallVector<StorageLifetime> allocations;
  function.walk<WalkOrder::PreOrder>([&](memref::AllocaOp allocation) {
    uint64_t start = std::numeric_limits<uint64_t>::max();
    uint64_t finish = end[allocation];
    auto scope = [&](Operation *operation) {
      // An asynchronous use can complete outside its allocation's conditional.
      if (!allocation->getParentRegion()->isAncestor(operation->getParentRegion()))
        return operation;
      Operation *lifetime = operation;
      while (operation->getBlock() != allocation->getBlock()) {
        operation = operation->getParentOp();
        if (!isa<scf::IfOp>(operation))
          lifetime = operation;
      }
      return lifetime;
    };
    auto aliases = storage.aliases(allocation);
    if (!aliases.complete) {
      // An unknown escape cannot justify reuse. Keep the allocation live for
      // the entire invocation instead of silently dropping an unmodeled use.
      allocations.push_back({allocation, begin[function], end[function]});
      return;
    }
    for (Operation *user : aliases.users) {
      auto effects = storage.effects(user);
      if (effects.complete && effects.entries.empty() && !effects.ordered)
        continue;
      start = std::min(start, begin[scope(user)]);
      Operation *last = user;
      auto completion = storage.completionOfUse(user);
      if (succeeded(completion)) {
        last = *completion;
      } else {
        // A transfer may still be pending at the next loop iteration. Extending
        // only the end would permit an earlier lexical allocation to overwrite
        // it on that backedge. Without a proved completion, exclude all reuse.
        start = begin[function];
        finish = end[function];
        break;
      }
      finish = std::max(finish, end[scope(last)]);
    }
    if (start == std::numeric_limits<uint64_t>::max())
      start = begin[allocation];
    allocations.push_back({allocation, start, finish});
  });
  llvm::stable_sort(allocations, [](const StorageLifetime &a, const StorageLifetime &b) {
    return a.start < b.start;
  });
  return allocations;
}

} // namespace intent::dsa
