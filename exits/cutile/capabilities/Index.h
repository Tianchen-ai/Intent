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
  auto view = mlir::dyn_cast<mlir::BlockArgument>(lin->getOperand(0));
  if (!view || view.getOwner() != &fn.front() ||
      !mlir::isa<mlir::MemRefType>(view.getType()))
    return false;
  auto *rOp = lin->getOperand(1).getDefiningOp(),
       *cOp = lin->getOperand(2).getDefiningOp();
  if (!rOp || rOp->getName().getStringRef() != "ct_src.rows" || !cOp ||
      cOp->getName().getStringRef() != "ct_src.columns")
    return false;
  bool validColOrigin = false;
  if (auto *z = cOp->getOperand(0).getDefiningOp())
    validColOrigin = z->getName().getStringRef() == "ct_src.zero" ||
                     (z->getName().getStringRef() == "ct_src.pid" &&
                      z->getAttrOfType<mlir::IntegerAttr>("axis").getInt() == 1);
  else if (auto tileIv = mlir::dyn_cast<mlir::BlockArgument>(cOp->getOperand(0)))
    validColOrigin = tileIv.getArgNumber() == 0 &&
                     tileIv.getOwner()->getParentOp() &&
                     tileIv.getOwner()->getParentOp()->getName().getStringRef() ==
                         "ct_src.tiles";
  if (!validColOrigin)
    return false;
  mlir::SmallVector<int64_t> dims;
  if (auto proj = fn.getArgAttrOfType<mlir::DenseI64ArrayAttr>(
          view.getArgNumber(), "ct_src.shape")) {
    dims.assign(proj.asArrayRef().begin(), proj.asArrayRef().end());
  } else {
    size_t rawSlot = view.getArgNumber();
    if (auto src = fn.getArgAttrOfType<mlir::IntegerAttr>(rawSlot, "kir.source")) {
      if (src.getInt() < 0)
        return false;
      rawSlot = static_cast<size_t>(src.getInt());
    }
    if (rawSlot >= facts->arguments.size())
      return false;
    auto *obj = facts->arguments[rawSlot].getAsObject();
    auto *arr = obj ? obj->getArray("shape") : nullptr;
    if (!arr || arr->empty())
      return false;
    for (const auto &v : *arr) {
      auto d = v.getAsInteger();
      if (!d || *d <= 0)
        return false;
      dims.push_back(*d);
    }
  }
  if (dims.empty())
    return false;
  int64_t memberWidth = 1;
  if (auto owner = fn->getAttrOfType<mlir::IntegerAttr>("kir.owner_axis")) {
    for (size_t i = 0; i < dims.size(); ++i)
      if (int64_t(i) != owner.getInt() &&
          __builtin_mul_overflow(memberWidth, dims[i], &memberWidth))
        return false;
  } else {
    memberWidth = dims.back();
  }
  if (memberWidth <= 0 || memberWidth % 4 != 0)
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
