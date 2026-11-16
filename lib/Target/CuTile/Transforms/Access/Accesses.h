#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_ACCESSES_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_ACCESSES_H

#include "../NativeRewrite.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cutile {

class AccessFormSelection;

struct NativeAccessInputs {
  llvm::SmallVector<block::LoadOp> loads;
  llvm::SmallVector<block::GatherOp> gathers;
  llvm::SmallVector<block::StoreOp> stores;
  llvm::SmallVector<block::AtomicRMWOp> atomics;
};

mlir::LogicalResult formNativeAccesses(
    mlir::func::FuncOp kernel, const NativeAccessInputs &inputs, bool matrixCompute,
    NativeFormRewriter &rewriter);
mlir::LogicalResult formNativeLoads(
    mlir::func::FuncOp kernel, llvm::ArrayRef<block::LoadOp> loads,
    bool matrixCompute, AccessFormSelection &forms, NativeFormRewriter &rewriter);
mlir::LogicalResult formFragmentExtractions(
    mlir::func::FuncOp kernel, llvm::ArrayRef<block::GatherOp> gathers,
    NativeFormRewriter &rewriter);
bool requiresFragmentStorage(block::GatherOp gather);
mlir::LogicalResult formNativeAtomics(
    mlir::func::FuncOp kernel, llvm::ArrayRef<block::AtomicRMWOp> atomics,
    NativeFormRewriter &rewriter);
mlir::LogicalResult formNativeStores(
    mlir::func::FuncOp kernel, llvm::ArrayRef<block::StoreOp> stores,
    AccessFormSelection &forms, NativeFormRewriter &rewriter);

} // namespace intent::cutile

#endif
