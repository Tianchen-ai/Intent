#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl {
mlir::LogicalResult verifyRowComputation(mlir::func::FuncOp, mlir::Block &, mlir::ValueRange captures, unsigned domainIndex, llvm::StringRef dialect, bool canonical);
mlir::LogicalResult verifyLogicalRows(mlir::ModuleOp, llvm::StringRef dialect, bool canonical);
}
