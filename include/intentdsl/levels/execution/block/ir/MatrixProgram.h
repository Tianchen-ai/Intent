#pragma once
#include "intentdsl/levels/execution/block/ir/TileBuilder.h"
namespace intentdsl::block {
using ScaledPanelLowering = mlir::Value (*)(Tiles &, MatrixAccess &);
using MatrixPanelRead = mlir::Value (*)(Tiles &, MatrixAccess &, bool left,
                                        mlir::Value group, mlir::Value panel);
using WindowPanelLowering = mlir::Value (*)(Tiles &, MatrixAccess &,
                                            mlir::Value &batch,
                                            mlir::Value &chunk,
                                            mlir::Value &head);
using MatrixPanelWrite = bool (*)(Tiles &, MatrixAccess &, mlir::Value dst,
                                  mlir::Value val,
                                  llvm::ArrayRef<mlir::Value> batchCoords);
void expandMatrixProgram(mlir::func::FuncOp, llvm::StringRef unit,
                         ScaledPanelLowering scaled = nullptr,
                         MatrixPanelRead read = nullptr,
                         WindowPanelLowering window = nullptr,
                         MatrixPanelWrite write = nullptr,
                         bool flatRecordBitPlanes = false);
} // namespace intentdsl::block
