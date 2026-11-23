#pragma once
#include "intentdsl/levels/execution/block/ir/TileBuilder.h"
namespace intentdsl::block {
using ScaledPanelLowering = mlir::Value (*)(Tiles &, MatrixAccess &);
void expandMatrixProgram(mlir::func::FuncOp, llvm::StringRef unit,
                         ScaledPanelLowering scaled = nullptr);
} // namespace intentdsl::block
