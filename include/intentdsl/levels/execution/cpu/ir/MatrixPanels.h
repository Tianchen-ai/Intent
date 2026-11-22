#pragma once
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"

namespace intentdsl::cpu {
// An output rectangle is local to one task. Coordinates below describe the
// original capture at its origin, with the published element strides.
struct PanelView {
  unsigned slot;
  std::string dtype, origin;
  std::vector<int64_t> strides;
  bool writes;
};
struct MatrixPanels {
  explicit MatrixPanels(mlir::func::FuncOp);
  mlir::func::FuncOp fn;
  mlir::Operation &domain, &dot;
  mlir::DenseI64ArrayAttr shape, tile;
  std::vector<PanelView> views;
  std::string offsets, routes;
  int64_t multiplicity = 1;
  std::string dispatch(llvm::StringRef declarations, llvm::StringRef invoke,
                       bool returnsStatus);
};
} // namespace intentdsl::cpu
