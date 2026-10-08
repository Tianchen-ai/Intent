#ifndef INTENT_CPU_TRANSFORMS_STORAGE_MEMORYACCESS_H
#define INTENT_CPU_TRANSFORMS_STORAGE_MEMORYACCESS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace intent::cpu::detail {

// One complete, unmasked scalar or vector access. The value type is part of
// its footprint; a vector<1xT> and a scalar T are distinct representations.
struct MemoryAccess {
  mlir::Operation *operation;
  mlir::Value memory;
  llvm::SmallVector<mlir::Value> indices;
  mlir::Value value;
  bool write;
};

std::optional<MemoryAccess> memoryAccess(mlir::Operation *operation);
bool sameMemoryValue(mlir::Value first, mlir::Value second,
                     const mlir::IRMapping &mapping);
bool sameMemoryAccess(const MemoryAccess &first, const MemoryAccess &second,
                      const mlir::IRMapping &mapping);
bool placeInvariantMemory(mlir::func::FuncOp function);

} // namespace intent::cpu::detail
#endif
