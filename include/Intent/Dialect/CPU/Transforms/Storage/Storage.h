#ifndef INTENT_DIALECT_CPU_TRANSFORMS_STORAGE_STORAGE_H
#define INTENT_DIALECT_CPU_TRANSFORMS_STORAGE_STORAGE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::memref { class CopyOp; }

namespace intent::cpu {

void eraseDeadPrivateBuffers(mlir::func::FuncOp function);
mlir::LogicalResult reusePrivateStorage(mlir::func::FuncOp function);
enum class ScratchRepresentation { PreserveDescriptors, LinearCapacity };
mlir::LogicalResult reuseScratchStorage(mlir::func::FuncOp function,
                                       ScratchRepresentation representation);
bool reuseMemoryValues(mlir::func::FuncOp function);
mlir::LogicalResult optimizeMemoryAccesses(mlir::func::FuncOp function);
void forwardCPUOutputs(mlir::func::FuncOp function);
mlir::LogicalResult fuseIntermediateBuffers(
    mlir::func::FuncOp function,
    llvm::function_ref<bool(mlir::memref::CopyOp)> retainCopy = nullptr);

} // namespace intent::cpu
#endif
