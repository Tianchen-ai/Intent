#ifndef INTENT_DIALECT_Block_TRANSFORMS_CONTRACTION_H
#define INTENT_DIALECT_Block_TRANSFORMS_CONTRACTION_H

#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent::block::contraction {

// Provider consumers inspect the same current physical contraction relation.
bool hasRangeContractForm(ContractOp contract,
                          llvm::SmallVectorImpl<StoreOp> *stores = nullptr);
mlir::LogicalResult normalizeMatrixContractShapes(mlir::func::FuncOp kernel);

// Source algebra normalization is shared by contraction and pointwise grouping.
mlir::LogicalResult fuseMultiplyReductions(mlir::ModuleOp module);

} // namespace intent::block::contraction

#endif
