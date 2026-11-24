#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::cutile {
struct ArrayIndices {
  unsigned bits;
  bool permitTma;
  bool scaledMma;
};
inline bool quadAlignedUniform(mlir::Operation *op, mlir::func::FuncOp fn,
                               const intentdsl::CallFacts *facts) {
  if (!op || op->getName().getStringRef() != "ct_src.uniform" || !facts)
    return false;
  auto *lin = op->getOperand(1).getDefiningOp();
  if (!lin || lin->getName().getStringRef() != "ct_src.linear_index")
    return false;
  auto *rOp = lin->getOperand(1).getDefiningOp(),
       *cOp = lin->getOperand(2).getDefiningOp();
  if (!rOp || rOp->getName().getStringRef() != "ct_src.rows" || !cOp ||
      cOp->getName().getStringRef() != "ct_src.columns")
    return false;
  auto [m, n] = facts->geometry(fn);
  int64_t cols = static_cast<int64_t>(n);
  if (n <= 0 || static_cast<double>(cols) != n || cols % 4 != 0)
    return false;
  auto configs = fn->getAttrOfType<mlir::ArrayAttr>("block.configs");
  if (!configs || configs.empty())
    return false;
  return llvm::all_of(configs, [](mlir::Attribute row) {
    auto w = mlir::cast<mlir::DictionaryAttr>(row)
                 .getAs<mlir::IntegerAttr>("width")
                 .getInt();
    return w >= 4 && w % 4 == 0;
  });
}
} // namespace intentdsl::cutile
