#include "MemoryAccess.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::cpu::detail {

std::optional<MemoryAccess> memoryAccess(Operation *operation) {
  if (auto load = dyn_cast<memref::LoadOp>(operation))
    return MemoryAccess{load, load.getMemref(), llvm::to_vector(load.getIndices()),
                        load.getResult(), false};
  if (auto store = dyn_cast<memref::StoreOp>(operation))
    return MemoryAccess{store, store.getMemref(), llvm::to_vector(store.getIndices()),
                        store.getValue(), true};
  if (auto load = dyn_cast<vector::LoadOp>(operation))
    return MemoryAccess{load, load.getBase(), llvm::to_vector(load.getIndices()),
                        load.getResult(), false};
  if (auto store = dyn_cast<vector::StoreOp>(operation))
    return MemoryAccess{store, store.getBase(), llvm::to_vector(store.getIndices()),
                        store.getValueToStore(), true};
  return std::nullopt;
}

bool sameMemoryValue(Value first, Value second, const IRMapping &mapping) {
  first = mapping.lookupOrDefault(first);
  second = mapping.lookupOrDefault(second);
  if (first == second) return true;
  auto lhs = getConstantIntValue(first), rhs = getConstantIntValue(second);
  if (lhs && rhs) return first.getType() == second.getType() && *lhs == *rhs;
  auto firstResult = dyn_cast<OpResult>(first), secondResult = dyn_cast<OpResult>(second);
  if (!firstResult || !secondResult ||
      firstResult.getResultNumber() != secondResult.getResultNumber()) return false;
  Operation *a = firstResult.getOwner(), *b = secondResult.getOwner();
  if (a->getNumRegions() || b->getNumRegions() ||
      !isMemoryEffectFree(a) || !isMemoryEffectFree(b)) return false;
  return OperationEquivalence::isEquivalentTo(a, b,
      [&](Value left, Value right) { return success(sameMemoryValue(left, right, mapping)); },
      [](Value, Value) {}, OperationEquivalence::IgnoreLocations);
}

bool sameMemoryAccess(const MemoryAccess &first, const MemoryAccess &second,
                      const IRMapping &mapping) {
  if (first.value.getType() != second.value.getType() ||
      first.indices.size() != second.indices.size() ||
      !sameMemoryValue(first.memory, second.memory, mapping)) return false;
  return llvm::all_of(llvm::zip(first.indices, second.indices), [&](auto pair) {
    return sameMemoryValue(std::get<0>(pair), std::get<1>(pair), mapping);
  });
}

} // namespace intent::cpu::detail
