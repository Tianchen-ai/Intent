#pragma once
#include "intentdsl/levels/algorithm/kir/ir/ContractionShape.h"
#include "intentdsl/levels/algorithm/kir/ir/Matrix.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"

namespace intentdsl::block {
// Complete microarchitecture-specific rows are contributed by the exit.
struct MatrixTile {
  int64_t m, n, k;
  std::string native;
  std::string operation = {};
  int64_t maxOperandBytes = 0;
  bool (*accepts)(mlir::Operation *domain, int64_t depth,
                  const CallFacts &facts) = nullptr;
};
struct MatrixTarget {
  std::vector<MatrixTile> tiles;
  std::string estimate;
  std::map<std::string, std::vector<std::string>> implementations;
  bool flatRecordBitPlanes = false;
};
const MatrixTile *findMatrixTile(mlir::Operation *, llvm::ArrayRef<int64_t>,
                                 const compilerinfra::TargetProfile &,
                                 std::optional<llvm::StringRef> native = std::nullopt);
bool supportsMatrixTile(mlir::Operation *, mlir::Attribute,
                        const compilerinfra::TargetProfile &);

llvm::Expected<kir::ContractionShape> matrixShape(mlir::Operation *domain,
                                                  const CallFacts &);
using VerifyPanelSetup =
    llvm::function_ref<mlir::LogicalResult(mlir::Operation &)>;
mlir::LogicalResult verifyTiledMatrix(mlir::func::FuncOp,
                                      const compilerinfra::TargetProfile &,
                                      llvm::StringRef dialect);
mlir::LogicalResult verifyTiledMatrix(mlir::func::FuncOp,
                                      const compilerinfra::TargetProfile &,
                                      llvm::StringRef dialect,
                                      VerifyPanelSetup);
llvm::Error emitMatrixPackage(mlir::func::FuncOp, llvm::StringRef source,
                              llvm::raw_pwrite_stream &, unsigned version,
                              llvm::json::Object nativeViews = {});
} // namespace intentdsl::block
