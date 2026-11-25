#pragma once
#include "intentdsl/levels/execution/block/ir/MatrixAccess.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include <optional>
#include <string>
namespace intentdsl::dsa {
struct MatrixStorage {
  int64_t (*privateBytes)(int64_t rows, int64_t columns, int64_t depth,
                          llvm::StringRef computation);
  int64_t wramBytes = 0;
  int64_t sramBytes = 0;
  int64_t clusters = 0;
  int64_t coresPerCluster = 0;
  std::string libraryF16Impl;
  std::string libraryFtzImpl;
  std::string libraryEstimator;
  // Native placement supported for this tile and worker count. Shared-LHS
  // tiles also have an independent private implementation.
  llvm::StringRef (*panelPlacement)(block::MatrixAccess &, int64_t,
                                    const MatrixStorage &);
};
struct PanelPlan {
  int64_t participants;
  llvm::StringRef placement;
  std::string id() const;
  mlir::DictionaryAttr attribute(mlir::Builder &) const;
  static std::optional<PanelPlan> read(mlir::Attribute, block::MatrixAccess &,
                                       const MatrixStorage &);
};
struct LibraryMatrixPlan {
  std::string impl;
  unsigned lhsSlot = 0, rhsSlot = 0, dstSlot = 0;
  int64_t batch = 1, m = 0, n = 0, k = 0;
  int32_t transA = 0, transB = 0;
  int64_t lhsBatchStride = 0, rhsBatchStride = 0, dstBatchStride = 0;
  int32_t inDtype = 0, outDtype = 0;
};
std::optional<LibraryMatrixPlan>
matchLibraryMatrix(block::MatrixAccess &, const compilerinfra::TargetProfile &);
mlir::LogicalResult verifyMatrix(mlir::func::FuncOp,
                                 const compilerinfra::TargetProfile &,
                                 bool supplied);
}
