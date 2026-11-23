#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::block {
// Lower one established row operation to symbolic row/column fragments.
// The exit supplies its native collective implementation namespace.
mlir::LogicalResult verifyRowProgram(mlir::func::FuncOp,
                                     llvm::StringRef dialect,
                                     llvm::StringRef helperAttr);
void expandRowOperation(mlir::Operation &, llvm::StringRef unit,
                        bool nativeHistogram);
} // namespace intentdsl::block
