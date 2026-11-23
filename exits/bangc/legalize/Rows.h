#pragma once

#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace CI_UNIT_NAMESPACE {
void rowContracts(compilerinfra::Unit &);
void addressOptimization(compilerinfra::Unit &);
bool canNarrowAddress(mlir::Operation &, mlir::func::FuncOp);
mlir::LogicalResult lowerRowsToCallable(mlir::ModuleOp,
                                        const compilerinfra::TargetProfile &);
mlir::LogicalResult verifyCallableRows(mlir::ModuleOp,
                                       const compilerinfra::TargetProfile &);
llvm::Error describeRowArtifact(mlir::func::FuncOp, llvm::raw_pwrite_stream &);
compilerinfra::TerminalPrinter rowPrinter();
} // namespace CI_UNIT_NAMESPACE
