#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::cpu {
struct RowConfig {
  int64_t grain, columns;
};
struct RowMachine {
  int64_t vectorBits, workers;
  unsigned packedOwners = 0;
  llvm::SmallVector<RowConfig> rows;
  bool structuredRows = false;
  bool boundWidening = false;
  int64_t wideningPanel =
      0; // Fixed-width panels supported for mixed-width live values.
  mlir::ArrayAttr wideningConfigurations(mlir::MLIRContext *ctx) const;
  mlir::ArrayAttr memberConfigurations(mlir::MLIRContext *ctx,
                                       bool widenOnly = false) const;
  mlir::ArrayAttr configurations(mlir::MLIRContext *, bool whole) const;
  bool accepts(mlir::ArrayAttr) const;
};
mlir::LogicalResult verifyTask(mlir::ModuleOp,
                               const compilerinfra::TargetProfile &,
                               llvm::StringRef storage,
                               llvm::StringRef taskOp = "cpu.task",
                               bool logicalBody = true);
llvm::Error realizeLocalValues(mlir::Operation *, mlir::Attribute,
                               const compilerinfra::TargetProfile &,
                               mlir::RewriterBase &,
                               llvm::StringRef realization = "private-values");
} // namespace intentdsl::cpu
