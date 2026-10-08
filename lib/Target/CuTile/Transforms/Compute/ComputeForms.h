#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_COMPUTE_COMPUTEFORMS_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_COMPUTE_COMPUTEFORMS_H

#include "../NativeRewrite.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cutile {

struct NativeComputeInputs {
  llvm::SmallVector<gpu::ContractOp> contracts;
  llvm::SmallVector<gpu::ScaledContractOp> scaledContracts;
  llvm::SmallVector<gpu::ReduceOp> reductions;
  llvm::SmallVector<gpu::HistogramOp> histograms;
  llvm::SmallVector<gpu::ScanOp> scans;
};

mlir::LogicalResult formComputePrimitives(
    mlir::func::FuncOp kernel, const NativeComputeInputs &inputs,
    NativeFormRewriter &rewriter);

mlir::LogicalResult formScaledMMA(mlir::func::FuncOp kernel,
    gpu::ScaledContractOp contract, NativeFormRewriter &rewriter);

} // namespace intent::cutile

#endif
