#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_COMPUTE_COMPUTEFORMS_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_COMPUTE_COMPUTEFORMS_H

#include "../NativeRewrite.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cutile {

struct NativeComputeInputs {
  llvm::SmallVector<block::ContractOp> contracts;
  llvm::SmallVector<block::ScaledContractOp> scaledContracts;
  llvm::SmallVector<block::ReduceOp> reductions;
  llvm::SmallVector<block::ScanOp> scans;
};

mlir::LogicalResult materializeHistograms(mlir::func::FuncOp kernel);

mlir::LogicalResult formComputePrimitives(
    mlir::func::FuncOp kernel, const NativeComputeInputs &inputs,
    NativeFormRewriter &rewriter);

mlir::LogicalResult formScaledMMA(mlir::func::FuncOp kernel,
    block::ScaledContractOp contract, NativeFormRewriter &rewriter);

mlir::LogicalResult formMMA(mlir::func::FuncOp kernel,
    block::ContractOp contract, NativeFormRewriter &rewriter);

} // namespace intent::cutile

#endif
