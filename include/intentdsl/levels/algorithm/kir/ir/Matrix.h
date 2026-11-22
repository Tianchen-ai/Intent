#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intentdsl::kir {
bool matrixDomain(mlir::func::FuncOp fn);
// A physical node may expose the logical calculation enclosed by its verified
// reduction loop. Its result occupies the loop's SSA result in the epilogue.
using MatrixExpression =
    llvm::function_ref<mlir::Operation *(mlir::Operation &)>;
mlir::LogicalResult verifyMatrix(mlir::func::FuncOp fn, llvm::StringRef dialect,
                                 bool canonical,
                                 MatrixExpression expression = nullptr);
mlir::LogicalResult verifyMatrixBody(mlir::func::FuncOp, mlir::Operation &,
                                     llvm::StringRef, bool,
                                     MatrixExpression = nullptr);
} // namespace intentdsl::kir
