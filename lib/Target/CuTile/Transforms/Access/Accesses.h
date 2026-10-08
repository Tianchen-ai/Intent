#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_ACCESSES_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_ACCESSES_H

#include "../NativeRewrite.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cutile {

class AccessFormSelection;

struct NativeAccessInputs {
  llvm::SmallVector<gpu::LoadOp> loads;
  llvm::SmallVector<gpu::GatherOp> gathers;
  llvm::SmallVector<gpu::StoreOp> stores;
  llvm::SmallVector<gpu::AtomicRMWOp> atomics;
};

mlir::LogicalResult formNativeAccesses(
    mlir::func::FuncOp kernel, const NativeAccessInputs &inputs, bool matrixCompute,
    NativeFormRewriter &rewriter);
mlir::LogicalResult formNativeLoads(
    mlir::func::FuncOp kernel, llvm::ArrayRef<gpu::LoadOp> loads,
    bool matrixCompute, AccessFormSelection &forms, NativeFormRewriter &rewriter);
mlir::LogicalResult formFragmentExtractions(
    mlir::func::FuncOp kernel, llvm::ArrayRef<gpu::GatherOp> gathers,
    NativeFormRewriter &rewriter);
bool requiresFragmentStorage(gpu::GatherOp gather);
mlir::LogicalResult formNativeAtomics(
    mlir::func::FuncOp kernel, llvm::ArrayRef<gpu::AtomicRMWOp> atomics,
    NativeFormRewriter &rewriter);
mlir::LogicalResult formNativeStores(
    mlir::func::FuncOp kernel, llvm::ArrayRef<gpu::StoreOp> stores,
    AccessFormSelection &forms, NativeFormRewriter &rewriter);

} // namespace intent::cutile

#endif
