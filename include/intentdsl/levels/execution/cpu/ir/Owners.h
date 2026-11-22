#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
namespace intentdsl::cpu {
bool ownerTask(mlir::func::FuncOp);
mlir::LogicalResult verifyOwnerTask(mlir::func::FuncOp,
                                    const compilerinfra::TargetProfile &,
                                    llvm::StringRef storage);
mlir::LogicalResult verifyOwnerGrain(mlir::func::FuncOp, mlir::Operation &,
                                     const compilerinfra::TargetProfile &);
std::string ownerDispatcher(mlir::func::FuncOp, llvm::StringRef declaration,
                            llvm::StringRef invocation);
} // namespace intentdsl::cpu
