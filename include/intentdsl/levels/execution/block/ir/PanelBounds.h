#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <algorithm>
#include <climits>
namespace intentdsl::block {
struct PanelRange {
  int64_t lo, hi;
};
// Signed range proof for the i32 panel trip count without 64-bit wraparound.
inline std::optional<PanelRange> panelRange(mlir::Value v,
                                            mlir::Operation &space,
                                            const intentdsl::CallFacts &facts) {
  using namespace mlir;
  constexpr __int128 kSafeMin = static_cast<__int128>(INT64_MIN) + 4096;
  constexpr __int128 kSafeMax = static_cast<__int128>(INT64_MAX) - 4096;
  auto fit = [&](__int128 lo, __int128 hi) -> std::optional<PanelRange> {
    if (lo > hi || lo < kSafeMin || hi > kSafeMax)
      return {};
    return PanelRange{int64_t(lo), int64_t(hi)};
  };
  auto floorDiv = [](__int128 a, __int128 d) -> __int128 {
    __int128 q = a / d, r = a % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
  };
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return fit(c.value(), c.value());
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue());
        ia && ia.getValue().getBitWidth() <= 64) {
      int64_t val = ia.getValue().getSExtValue();
      return fit(val, val);
    }
  if (auto dim = v.getDefiningOp<memref::DimOp>()) {
    auto shape = facts.shapeOf(dim.getSource());
    auto ax = dim.getConstantIndex();
    if (shape && ax) {
      auto n = (*shape)[*ax].getAsInteger();
      if (n && *n >= 0 && *n <= INT32_MAX)
        return PanelRange{*n, *n};
    }
    return {};
  }
  if (auto i = dyn_cast<BlockArgument>(v);
      i && i.getOwner() == &space.getRegion(0).front()) {
    auto n = i.getArgNumber();
    auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
    auto sh = facts.shapeOf(space.getOperand(n));
    if (sh && axes) {
      auto ext = (*sh)[axes[n]].getAsInteger();
      if (ext && *ext >= 0 && *ext <= INT32_MAX)
        return PanelRange{0, std::max<int64_t>(0, *ext - 1)};
    }
    return {};
  }
  auto *p = v.getDefiningOp();
  if (!p)
    return {};
  if (auto ic = dyn_cast<arith::IndexCastOp>(p)) {
    if (auto it = dyn_cast<IntegerType>(ic.getIn().getType());
        it && it.isSignless() && it.getWidth() > 0 && it.getWidth() <= 32) {
      unsigned w = it.getWidth();
      return PanelRange{-(int64_t(1) << (w - 1)), (int64_t(1) << (w - 1)) - 1};
    }
    return {};
  }
  if (auto br = dyn_cast<scf::IfOp>(p);
      br && br.getNumResults() == 1 && !br.getElseRegion().empty()) {
    auto a = panelRange(br.thenYield().getOperand(0), space, facts),
         b = panelRange(br.elseYield().getOperand(0), space, facts);
    if (a && b)
      return PanelRange{std::min(a->lo, b->lo), std::max(a->hi, b->hi)};
    return {};
  }
  if (auto sel = dyn_cast<arith::SelectOp>(p)) {
    auto a = panelRange(sel.getTrueValue(), space, facts),
         b = panelRange(sel.getFalseValue(), space, facts);
    if (a && b)
      return PanelRange{std::min(a->lo, b->lo), std::max(a->hi, b->hi)};
    return {};
  }
  if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::FloorDivSIOp,
           arith::MinSIOp, arith::MaxSIOp>(p))
    return {};
  auto a = panelRange(p->getOperand(0), space, facts),
       b = panelRange(p->getOperand(1), space, facts);
  if (!a || !b)
    return {};
  __int128 al = a->lo, ah = a->hi, bl = b->lo, bh = b->hi;
  if (isa<arith::MinSIOp>(p))
    return fit(std::min(al, bl), std::min(ah, bh));
  if (isa<arith::MaxSIOp>(p))
    return fit(std::max(al, bl), std::max(ah, bh));
  if (isa<arith::FloorDivSIOp>(p)) {
    if (p->getOperand(1).getDefiningOp<arith::ConstantIndexOp>() && bl > 0 &&
        bl == bh)
      return fit(floorDiv(al, bl), floorDiv(ah, bl));
    return {};
  }
  if (isa<arith::AddIOp>(p))
    return fit(al + bl, ah + bh);
  if (isa<arith::SubIOp>(p))
    return fit(al - bh, ah - bl);
  auto extremes = std::minmax({al * bl, al * bh, ah * bl, ah * bh});
  return fit(extremes.first, extremes.second);
}
inline bool countedPanel(mlir::Operation *o,
                         const intentdsl::CallFacts &facts) {
  auto l = mlir::dyn_cast<mlir::scf::ForOp>(o);
  if (!l || !o->hasAttr("block.uniform_tokens"))
    return false;
  int64_t span = 1;
  if (auto cw = o->getAttrOfType<mlir::IntegerAttr>("block.contribution_width"))
    span = std::max<int64_t>(1, cw.getInt());
  else if (auto *parent = o->getParentOp())
    if (auto ts = parent->getAttrOfType<mlir::IntegerAttr>("block.token_span"))
      span = std::max<int64_t>(1, ts.getInt());
  auto lb = panelRange(l.getLowerBound(), *o->getParentOp(), facts),
       ub = panelRange(l.getUpperBound(), *o->getParentOp(), facts);
  if (!lb || !ub)
    return false;
  auto floorDiv = [](__int128 a, __int128 d) -> __int128 {
    __int128 q = a / d, r = a % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
  };
  __int128 bLo = floorDiv(lb->lo, span), bHi = floorDiv(lb->hi, span);
  __int128 eLo = floorDiv(static_cast<__int128>(ub->lo) + (span - 1), span),
           eHi = floorDiv(static_cast<__int128>(ub->hi) + (span - 1), span);
  return bLo >= INT32_MIN && bHi <= INT32_MAX && eLo >= INT32_MIN &&
         eHi <= INT32_MAX;
}
} // namespace intentdsl::block
