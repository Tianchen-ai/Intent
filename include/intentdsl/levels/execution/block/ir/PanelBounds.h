#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace intentdsl::block {
// Range proof for the signed i32 panel trip count. Tensor addressing still
// uses index arithmetic. A data-dependent interval keeps the wide loop.
inline std::optional<int64_t> panelBound(mlir::Value v, mlir::Operation &space,
                                         const intentdsl::CallFacts &facts) {
  using namespace mlir;
  auto checked = [](int64_t n) -> std::optional<int64_t> {
    if (n >= 0 && n <= INT32_MAX)
      return n;
    return {};
  };
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return checked(c.value());
  if (auto dim = v.getDefiningOp<memref::DimOp>()) {
    auto shape = facts.shapeOf(dim.getSource());
    auto ax = dim.getConstantIndex();
    if (shape && ax)
      return checked((*shape)[*ax].getAsInteger().value_or(-1));
    return {};
  }
  if (auto i = dyn_cast<BlockArgument>(v);
      i && i.getOwner() == &space.getRegion(0).front()) {
    auto n = i.getArgNumber();
    auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
    auto sh = facts.shapeOf(space.getOperand(n));
    if (sh && axes)
      return checked((*sh)[axes[n]].getAsInteger().value_or(0) - 1);
    return {};
  }
  auto *p = v.getDefiningOp();
  if (auto br = dyn_cast_if_present<scf::IfOp>(p);
      br && br.getNumResults() == 1 && !br.getElseRegion().empty()) {
    auto a = panelBound(br.thenYield().getOperand(0), space, facts),
         b = panelBound(br.elseYield().getOperand(0), space, facts);
    if (a && b)
      return std::max(*a, *b);
    return {};
  }
  if (!p ||
      !isa<arith::AddIOp, arith::MulIOp, arith::FloorDivSIOp, arith::MinSIOp>(
          p))
    return {};
  auto a = panelBound(p->getOperand(0), space, facts),
       b = panelBound(p->getOperand(1), space, facts);
  if (!a || !b)
    return {};
  if (isa<arith::MinSIOp>(p))
    return std::min(*a, *b);
  if (isa<arith::FloorDivSIOp>(p)) {
    if (p->getOperand(1).getDefiningOp<arith::ConstantIndexOp>() && *b > 0)
      return *a / *b;
    return {};
  }
  return checked(isa<arith::AddIOp>(p) ? *a + *b : *a * *b);
}
inline bool countedPanel(mlir::Operation *o,
                         const intentdsl::CallFacts &facts) {
  auto l = mlir::dyn_cast<mlir::scf::ForOp>(o);
  return l && o->hasAttr("block.uniform_tokens") &&
         panelBound(l.getLowerBound(), *o->getParentOp(), facts) &&
         panelBound(l.getUpperBound(), *o->getParentOp(), facts);
}
} // namespace intentdsl::block
