#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::cpu {
struct RowMachine {
  int64_t vectorBits, workers;
  llvm::SmallVector<int64_t> rows;
};
mlir::LogicalResult verifyTask(mlir::ModuleOp,
                               const compilerinfra::TargetProfile &,
                               llvm::StringRef storage,
                               llvm::StringRef taskOp = "cpu.task",
                               bool logicalBody = true);
llvm::Error realizeLocalValues(mlir::Operation *, mlir::Attribute,
                               const compilerinfra::TargetProfile &,
                               mlir::RewriterBase &);
} // namespace intentdsl::cpu
