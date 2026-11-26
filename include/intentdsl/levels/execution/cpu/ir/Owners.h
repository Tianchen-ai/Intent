#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
namespace intentdsl::cpu {
bool ownerTask(mlir::func::FuncOp);
mlir::LogicalResult verifyOwnerTask(mlir::func::FuncOp,
                                    const compilerinfra::TargetProfile &,
                                    llvm::StringRef storage);
mlir::LogicalResult verifyOwnerBuffers(
    mlir::func::FuncOp, const compilerinfra::TargetProfile &,
    llvm::StringRef realization = "private-values");
llvm::Error materializeOwnerBuffers(
    mlir::func::FuncOp, mlir::RewriterBase &, const compilerinfra::TargetProfile &,
    llvm::StringRef realization = "private-values");
void screenRankedInsertions(mlir::Block &, mlir::RewriterBase &,
                            const compilerinfra::TargetProfile &);
mlir::LogicalResult verifyOwnerGrain(mlir::func::FuncOp, mlir::Operation &,
                                     const compilerinfra::TargetProfile &);
} // namespace intentdsl::cpu
