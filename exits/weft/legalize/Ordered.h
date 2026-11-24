#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace CI_UNIT_NAMESPACE {
mlir::LogicalResult bindWeftMemberStores(mlir::Operation &task, bool rvv,
                                         bool assign);
mlir::LogicalResult lowerWeftOrdered(mlir::func::FuncOp fn);
bool hasWeftOwnerProgram(mlir::ModuleOp m);
llvm::Error describeWeftOrderedPackage(mlir::func::FuncOp fn,
                                       mlir::Operation &prog,
                                       llvm::raw_ostream &os);
mlir::LogicalResult verifyWeftOrdered(mlir::ModuleOp m,
                                      const compilerinfra::TargetProfile &p,
                                      bool callable);
} // namespace CI_UNIT_NAMESPACE
