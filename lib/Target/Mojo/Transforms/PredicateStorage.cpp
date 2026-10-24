#include "Legalize.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace intent::mojo {

void materializePredicateStorage(func::FuncOp function) {
  // Logical predicates remain i1. Mojo's independently addressable bool slots
  // occupy bytes; make that private representation explicit so vector loads do
  // not accidentally request packed i1 memory or block a whole SIMD producer.
  SmallVector<Operation *> allocations;
  function.walk([&](Operation *operation) {
    if (isa<memref::AllocOp, memref::AllocaOp>(operation) &&
        cast<MemRefType>(operation->getResult(0).getType()).getElementType().isInteger(1))
      allocations.push_back(operation);
  });
  for (Operation *allocation : allocations) {
    Value memory = allocation->getResult(0);
    cpu::StorageAnalysis storage(function);
    auto aliases = storage.aliases(memory);
    if (!aliases.complete || !llvm::all_of(aliases.values, [&](Value value) {
          auto type = dyn_cast<MemRefType>(value.getType());
          return type && type.getElementType().isInteger(1) && storage.uniqueOrigin(value) == memory;
        })) continue;
    if (!llvm::all_of(aliases.users, [](Operation *user) {
          return cpu::isStorageAliasOperation(user) ||
                 isa<memref::LoadOp, memref::StoreOp, memref::DimOp, memref::DeallocOp>(user);
        })) continue;
    OpBuilder builder(allocation);
    Type byte = builder.getI8Type();
    for (Value value : aliases.values) {
      auto type = cast<MemRefType>(value.getType());
      value.setType(MemRefType::get(type.getShape(), byte, type.getLayout(), type.getMemorySpace()));
    }
    for (Operation *user : aliases.users) {
      if (auto load = dyn_cast<memref::LoadOp>(user)) {
        load.getResult().setType(byte);
        builder.setInsertionPointAfter(load);
        auto predicate = builder.create<arith::TruncIOp>(load.getLoc(), builder.getI1Type(), load.getResult());
        load.getResult().replaceAllUsesExcept(predicate.getResult(), predicate);
      } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
        builder.setInsertionPoint(store);
        Value value = builder.create<arith::ExtUIOp>(store.getLoc(), byte, store.getValue());
        store.getValueMutable().assign(value);
      }
    }
  }
}

} // namespace intent::mojo
