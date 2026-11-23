#pragma once
#include "intentdsl/levels/execution/block/ir/TileBuilder.h"
namespace intentdsl::block {
using ScaledPanelLowering = mlir::Value (*)(Tiles &, MatrixAccess &);
using MatrixPanelRead = mlir::Value (*)(Tiles &, MatrixAccess &, bool left,
                                        mlir::Value group, mlir::Value panel);
void expandMatrixProgram(mlir::func::FuncOp, llvm::StringRef unit,
                         ScaledPanelLowering scaled = nullptr,
                         MatrixPanelRead read = nullptr);
} // namespace intentdsl::block
