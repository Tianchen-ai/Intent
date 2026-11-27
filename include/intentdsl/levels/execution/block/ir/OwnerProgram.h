#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace compilerinfra {
struct TargetProfile;
}
namespace intentdsl::block {
mlir::LogicalResult
verifyTileProgram(mlir::func::FuncOp, const compilerinfra::TargetProfile &,
                  llvm::function_ref<bool(mlir::Operation &)> provesNarrow);
struct Tiles;
struct OwnerAccess {
  // Return false when elementwise coordinates must be retained.
  bool (*rectangle)(Tiles &, mlir::Operation &, llvm::ArrayRef<mlir::Value>,
                    int64_t rows, mlir::Value stored,
                    mlir::Value &result) = nullptr;
  mlir::Value (*panel)(Tiles &, mlir::Operation &, llvm::ArrayRef<mlir::Value>,
                       mlir::Value start, int64_t span,
                       int64_t width) = nullptr;
  bool scalarBroadcastRead = false;
  bool nativeTileGather = false;
  bool nativeTileHistogram = false;
  bool blockAtomicMemoryBins = false;
  int64_t minMmaRows = 0;
  bool (*groupedPanel)(Tiles &, mlir::Value view,
                       llvm::ArrayRef<mlir::Value> coords, int64_t panelAxis,
                       mlir::Value panelBase, int64_t panelSpan,
                       int64_t sliceAxis, int64_t sliceWidth, int64_t rows,
                       llvm::SmallVectorImpl<mlir::Value> &slices) = nullptr;
  int64_t scatterPanelWidth = 0;
  int64_t histogramPanelWidth = 0;
  int64_t minHistogramPanelWidth = 0;
  int64_t scatterFragmentElements = 0;
};
void expandOwnerProgram(mlir::func::FuncOp fn, llvm::StringRef unit,
                        OwnerAccess access = {});
} // namespace intentdsl::block
