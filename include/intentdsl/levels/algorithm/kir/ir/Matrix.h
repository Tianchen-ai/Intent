#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intentdsl::kir {
bool matrixDomain(mlir::func::FuncOp fn);
mlir::LogicalResult verifyMatrix(mlir::func::FuncOp fn, llvm::StringRef dialect, bool canonical);
}
