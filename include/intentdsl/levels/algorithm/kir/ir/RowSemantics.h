#pragma once
#include "mlir/IR/BuiltinOps.h"
namespace intentdsl {
mlir::LogicalResult verifyLogicalRows(mlir::ModuleOp, llvm::StringRef dialect, bool canonical);
}
