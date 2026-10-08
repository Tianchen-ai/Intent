#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

Operation *lastPayloadUse(StoreOp store) {
  Operation *last = store;
  SmallVector<Value> pending{store.getValue()};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (user == store)
        continue;
      Operation *ancestor = store->getBlock()->findAncestorOpInBlock(*user);
      if (ancestor && last->isBeforeInBlock(ancestor))
        last = ancestor;
      if (isa<BroadcastOp, SplatOp, ReshapeOp, TransposeOp, CastOp>(user))
        pending.append(user->getResults().begin(), user->getResults().end());
    }
  }
  return last;
}

bool canCross(StoreOp store, Operation &operation) {
  if (isa<LoadOp, StoreOp>(operation))
    return haveDisjointPrivateBufferAccesses(store, &operation);
  if (isa<scf::ForOp, scf::IfOp>(operation)) {
    for (Region &region : operation.getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          if (!canCross(store, nested))
            return false;
    return true;
  }
  // Unknown regions, atomics, barriers and allocation/free effects stop motion.
  return operation.getNumRegions() == 0 && isPure(&operation) &&
         (!operation.hasTrait<OpTrait::IsTerminator>() ||
          isa<scf::YieldOp>(operation));
}

void schedule(Block &block) {
  SmallVector<StoreOp> stores;
  for (Operation &operation : block) {
    for (Region &region : operation.getRegions())
      for (Block &nested : region)
        schedule(nested);
    if (auto store = dyn_cast<StoreOp>(operation))
      stores.push_back(store);
  }
  for (StoreOp store : stores) {
    auto buffer = dyn_cast<BufferType>(store.getResource().getType());
    if (!buffer || buffer.getScope().getValue() != BufferScope::ProgramPrivate ||
        buffer.isInvocationWorkspace())
      continue;
    Operation *last = lastPayloadUse(store);
    if (!isa<scf::ForOp, scf::IfOp>(last))
      continue;
    bool disjoint = true;
    for (Operation *next = store->getNextNode(); next; next = next->getNextNode()) {
      if (!canCross(store, *next)) {
        disjoint = false;
        break;
      }
      if (next == last)
        break;
    }
    if (disjoint)
      store->moveAfter(last);
  }
}

} // namespace

LogicalResult schedulePrivateStores(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  schedule(kernel->front());
  return success();
}

} // namespace intent::gpu
