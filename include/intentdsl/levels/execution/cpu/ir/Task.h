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
  unsigned orderedMatrixOwners = 0;
  unsigned orderedValueMinColumns = 1, orderedValueColumnMultiple = 1;
  // Width of a downstream score contraction. Zero covers the complete key
  // panel.
  unsigned orderedDotTokens = 0;
  int64_t ownerStackBytes = 0, ownerRegisters = 0;
  int64_t l2CacheBytes = 0;
  static constexpr unsigned streamedDotTokens = 16;
  llvm::SmallVector<RowConfig> rows;
  llvm::SmallVector<int64_t, 2> unrollFactors = {1};
  llvm::SmallVector<int64_t, 2> segmentWidths = {8};
  std::string estimate;
  bool structuredRows = false;
  bool boundWidening = false;
  int64_t wideningPanel =
      0; // Fixed-width panels supported for mixed-width live values.
  mlir::ArrayAttr wideningConfigurations(mlir::MLIRContext *ctx) const;
  mlir::ArrayAttr memberConfigurations(mlir::MLIRContext *ctx,
                                       bool widenOnly = false) const;
  mlir::ArrayAttr configurations(mlir::MLIRContext *, bool whole) const;
  bool accepts(mlir::ArrayAttr) const;
  bool acceptsUnroll(int64_t factor) const {
    return llvm::is_contained(unrollFactors, factor);
  }
  bool acceptsSegmentWidth(int64_t width) const {
    return llvm::is_contained(segmentWidths, width);
  }
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
