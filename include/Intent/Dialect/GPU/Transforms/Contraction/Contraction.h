#ifndef INTENT_DIALECT_GPU_TRANSFORMS_CONTRACTION_H
#define INTENT_DIALECT_GPU_TRANSFORMS_CONTRACTION_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent::gpu::contraction {

// Provider consumers inspect the same current physical contraction relation.
bool hasRangeContractForm(ContractOp contract,
                          llvm::SmallVectorImpl<StoreOp> *stores = nullptr);
mlir::LogicalResult normalizeMatrixContractShapes(mlir::func::FuncOp kernel);

// Source algebra normalization is shared by contraction and pointwise grouping.
mlir::LogicalResult fuseMultiplyReductions(mlir::ModuleOp module);

} // namespace intent::gpu::contraction

#endif
