#pragma once
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace CI_UNIT_NAMESPACE {
mlir::LogicalResult bindWeftMemberStores(mlir::Operation &task, bool rvv,
                                         bool assign);
mlir::LogicalResult lowerWeftOrdered(mlir::func::FuncOp fn,
                                     const intentdsl::CallFacts &facts,
                                     bool rvv);
void lowerInvocationPanels(
    mlir::func::FuncOp fn, mlir::Operation &program, mlir::Operation &kernel,
    mlir::Operation &team, mlir::DenseI64ArrayAttr slots,
    const llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t>> &axes);
bool hasWeftOwnerProgram(mlir::ModuleOp m);
llvm::Error describeWeftOrderedPackage(mlir::func::FuncOp fn,
                                       mlir::Operation &prog,
                                       llvm::raw_ostream &os);
mlir::LogicalResult verifyWeftOrdered(mlir::ModuleOp m,
                                      const compilerinfra::TargetProfile &p,
                                      bool callable);
} // namespace CI_UNIT_NAMESPACE
