#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Matrix.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/algorithm/kir/ir/ContractionShape.h"

namespace intentdsl::block {
// Complete microarchitecture-specific rows are contributed by the exit.
struct MatrixTile {int64_t m,n,k; std::string native;};
struct MatrixTarget {std::vector<MatrixTile> tiles;};

llvm::Expected<kir::ContractionShape> matrixShape(mlir::Operation *domain,const CallFacts &);
using VerifyPanelSetup = llvm::function_ref<mlir::LogicalResult(mlir::Operation &)>;
mlir::LogicalResult verifyTiledMatrix(mlir::func::FuncOp, const compilerinfra::TargetProfile &, llvm::StringRef dialect);
mlir::LogicalResult verifyTiledMatrix(mlir::func::FuncOp, const compilerinfra::TargetProfile &, llvm::StringRef dialect, VerifyPanelSetup);
llvm::Error emitMatrixPackage(mlir::func::FuncOp,llvm::StringRef source,llvm::raw_pwrite_stream &,unsigned version,llvm::json::Object nativeViews={});
}
