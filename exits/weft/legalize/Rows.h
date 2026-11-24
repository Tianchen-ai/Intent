#pragma once
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace CI_UNIT_NAMESPACE {
mlir::LogicalResult lowerWeftRow(mlir::func::FuncOp fn,
                                 const intentdsl::CallFacts &facts,
                                 bool grouped = false,
                                 int64_t tailVecWidth = 1,
                                 int64_t privateStackBytes = 65536);
bool hasWeftRowProgram(mlir::ModuleOp m);
llvm::Error describeWeftRowPackage(mlir::func::FuncOp fn, mlir::Operation &prog,
                                   llvm::raw_ostream &os);
mlir::LogicalResult verifyWeftRow(mlir::ModuleOp m,
                                  const compilerinfra::TargetProfile &p,
                                  bool callable);
} // namespace CI_UNIT_NAMESPACE
