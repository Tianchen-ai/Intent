#include "TaskLowering.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <functional>

using namespace mlir;

namespace intent::weft_provider {

LogicalResult materializeAddressableTaskStorage(func::FuncOp function) {
  cpu::StorageAnalysis storage(function);
  SmallVector<memref::AllocOp> allocations;
  bool unresolved = false;
  auto requireStorage = [&](Value memory) {
    auto origins = storage.origins(memory);
    unresolved |= !origins.complete;
    for (Value origin : origins.values)
      if (auto allocation = origin.getDefiningOp<memref::AllocOp>();
          allocation && allocation->getParentOfType<cpu::TasksOp>() &&
          !llvm::is_contained(allocations, allocation))
        allocations.push_back(allocation);
  };
  function.walk([&](cpu::QuantizeOp op) { requireStorage(op.getOutput()); });
  function.walk([&](cpu::QuantizedDotOp op) {
    requireStorage(op.getLhs());
    requireStorage(op.getRhs());
  });
  if (unresolved)
    return function.emitError("addressable task storage has unresolved owners");

  for (memref::AllocOp allocation : allocations) {
    auto task = allocation->getParentOfType<cpu::TasksOp>();
    Block &body = task.getBody().front();
    cpu::StorageAnalysis currentStorage(function);
    auto aliases = currentStorage.aliases(allocation);
    if (allocation->getBlock() != &body || !aliases.complete ||
        llvm::any_of(aliases.users, [&](Operation *user) {
          return !task->isProperAncestor(user);
        }))
      return allocation.emitError(
          "addressable private storage requires a nonescaping task-scope allocation");

    IRMapping mapping;
    for (auto [argument, capture] :
         llvm::zip_equal(body.getArguments().drop_front(), task.getCaptures()))
      mapping.map(argument, capture);
    SmallVector<Operation *> extentProducers;
    llvm::DenseSet<Operation *> visited;
    std::function<bool(Value)> available = [&](Value value) {
      if (mapping.contains(value)) return true;
      Operation *producer = value.getDefiningOp();
      if (!producer || !task->isProperAncestor(producer) ||
          producer->getNumRegions() || !isPure(producer))
        return false;
      if (!visited.insert(producer).second) return true;
      if (!llvm::all_of(producer->getOperands(), available)) return false;
      extentProducers.push_back(producer);
      return true;
    };
    if (!llvm::all_of(allocation.getDynamicSizes(), available))
      return allocation.emitError(
          "addressable private storage extents must be invariant across tasks");

    OpBuilder host(task);
    for (Operation *producer : extentProducers) host.clone(*producer, mapping);
    SmallVector<Value> dynamicSizes{task.getCount()};
    for (Value extent : allocation.getDynamicSizes())
      dynamicSizes.push_back(mapping.lookup(extent));
    auto original = allocation.getType();
    SmallVector<int64_t> shape{ShapedType::kDynamic};
    llvm::append_range(shape, original.getShape());
    auto backingType = MemRefType::get(shape, original.getElementType(),
        MemRefLayoutAttrInterface{}, original.getMemorySpace());
    auto backing = host.create<memref::AllocOp>(allocation.getLoc(), backingType,
        dynamicSizes, allocation.getAlignmentAttr());
    task.getCapturesMutable().append(ValueRange{backing.getResult()});
    Value capture = body.addArgument(backingType, allocation.getLoc());

    OpBuilder local(allocation);
    SmallVector<OpFoldResult> offsets{body.getArgument(0)};
    SmallVector<OpFoldResult> sizes{local.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides(shape.size(), local.getIndexAttr(1));
    unsigned dynamic = 0;
    for (int64_t extent : original.getShape()) {
      offsets.push_back(local.getIndexAttr(0));
      sizes.push_back(ShapedType::isDynamic(extent)
          ? OpFoldResult(allocation.getDynamicSizes()[dynamic++])
          : OpFoldResult(local.getIndexAttr(extent)));
    }
    auto sliceType = memref::SubViewOp::inferRankReducedResultType(
        original.getShape(), backingType, offsets, sizes, strides);
    Value slice = local.create<memref::SubViewOp>(allocation.getLoc(),
        cast<MemRefType>(sliceType), capture, offsets, sizes, strides);
    // The leading task coordinate makes private slices disjoint. One host owner
    // spans the task join; no encoded storage is reinterpreted as a numeric Value.
    for (Operation *user : aliases.users)
      if (isa<memref::DeallocOp>(user)) user->erase();
    allocation.replaceAllUsesWith(slice);
    allocation.erase();
    host.setInsertionPointAfter(task);
    host.create<memref::DeallocOp>(task.getLoc(), backing);
  }
  return success();
}

} // namespace intent::weft_provider
