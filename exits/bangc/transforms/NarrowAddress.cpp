#include "../legalize/Rows.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <climits>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;

namespace {
struct Interval {
  int64_t lo, hi;
};
using Range = std::optional<Interval>;

Range fit(Type ty, __int128 lo, __int128 hi) {
  if (lo > hi || lo < INT32_MIN || hi > INT32_MAX)
    return {};
  unsigned bits = 0;
  if (ty.isIndex())
    bits = 64;
  else if (auto it = dyn_cast<IntegerType>(ty); it && it.isSignless())
    bits = it.getWidth();
  if (bits < 32)
    return {};
  return Interval{int64_t(lo), int64_t(hi)};
}

struct SymbolBounds {
  int64_t minR = 1, maxR = 0;
  int64_t batch = 1;
  int64_t minW = 1, maxW = 0;
  int64_t minP = 1, maxP = 0;
  int64_t minB = 1, maxB = 0;
  bool valid = false;

  static SymbolBounds of(func::FuncOp fn) {
    SymbolBounds sb;
    auto configs = fn->getAttrOfType<ArrayAttr>("block.configs");
    auto layout = fn->getAttrOfType<DictionaryAttr>("dsa.layout");
    if (!configs || !layout || !layout.getAs<ArrayAttr>("rows") ||
        layout.getAs<ArrayAttr>("rows").empty())
      return sb;
    sb.batch = layout.getAs<IntegerAttr>("batch")
                   ? layout.getAs<IntegerAttr>("batch").getInt()
                   : 1;
    bool first = true;
    for (auto a : layout.getAs<ArrayAttr>("rows")) {
      auto alloc = cast<DictionaryAttr>(a);
      auto id = alloc.getAs<IntegerAttr>("id");
      auto p = alloc.getAs<IntegerAttr>("pitch").getInt();
      auto bytes = alloc.getAs<IntegerAttr>("bytes").getInt();
      for (auto ca : configs) {
        auto row = cast<DictionaryAttr>(ca);
        if (row.get("id") != id)
          continue;
        auto r = row.getAs<IntegerAttr>("rows").getInt();
        auto w = row.getAs<IntegerAttr>("width").getInt();
        if (first) {
          sb.minR = sb.maxR = r;
          sb.minW = sb.maxW = w;
          sb.minP = sb.maxP = p;
          sb.minB = sb.maxB = bytes;
          first = false;
        } else {
          sb.minR = std::min(sb.minR, r);
          sb.maxR = std::max(sb.maxR, r);
          sb.minW = std::min(sb.minW, w);
          sb.maxW = std::max(sb.maxW, w);
          sb.minP = std::min(sb.minP, p);
          sb.maxP = std::max(sb.maxP, p);
          sb.minB = std::min(sb.minB, bytes);
          sb.maxB = std::max(sb.maxB, bytes);
        }
      }
    }
    sb.valid = !first && sb.maxR <= INT32_MAX / std::max<int64_t>(1, sb.batch) &&
               sb.maxW <= INT32_MAX && sb.maxP <= INT32_MAX &&
               sb.maxB <= INT32_MAX;
    return sb;
  }
};

class Bounds {
  SymbolBounds sym;
  llvm::DenseMap<Value, Range> known;

  Range compute(Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      auto *parent = arg.getOwner()->getParentOp();
      if (auto loop = dyn_cast_or_null<scf::ForOp>(parent)) {
        if (loop.getInductionVar() == v) {
          auto a = get(loop.getLowerBound()), z = get(loop.getUpperBound()),
               s = get(loop.getStep());
          if (!a || !z || !s || s->lo <= 0)
            return {};
          return fit(v.getType(), a->lo, std::max<__int128>(a->lo, z->hi - 1));
        }
        unsigned idx = arg.getArgNumber() - 1;
        auto init = get(loop.getInitArgs()[idx]);
        if (!init)
          return {};
        Value yielded = loop.getBody()->getTerminator()->getOperand(idx);
        if (yielded == v)
          return fit(v.getType(), init->lo, init->hi);
        if (auto sub = yielded.getDefiningOp<arith::SubIOp>();
            sub && sub.getRhs() == v) {
          if (auto one = get(sub.getLhs());
              one && one->lo == 1 && one->hi == 1 && init->lo == 0 &&
              init->hi == 0)
            return fit(v.getType(), 0, 1);
        }
        auto next = get(yielded);
        if (!next)
          return {};
        return fit(v.getType(), std::min(init->lo, next->lo),
                   std::max(init->hi, next->hi));
      }
      return {};
    }
    auto *op = v.getDefiningOp();
    if (!op)
      return {};
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
        return fit(v.getType(), attr.getInt(), attr.getInt());
      return {};
    }
    if (op->getName().getStringRef() == "bangc_src.symbol" && sym.valid) {
      auto name = op->getAttrOfType<StringAttr>("name").getValue();
      if (name == "R")
        return fit(v.getType(), sym.minR, sym.maxR);
      if (name == "B")
        return fit(v.getType(), sym.batch, sym.batch);
      if (name == "EB")
        return fit(v.getType(), sym.minR * sym.batch, sym.maxR * sym.batch);
      if (name == "W")
        return fit(v.getType(), sym.minW, sym.maxW);
      if (name == "P")
        return fit(v.getType(), sym.minP, sym.maxP);
      if (name == "BYTES")
        return fit(v.getType(), sym.minB, sym.maxB);
      return {};
    }
    if (isa<arith::ExtSIOp>(op)) {
      auto r = get(op->getOperand(0));
      return r ? fit(v.getType(), r->lo, r->hi) : Range{};
    }
    if (isa<arith::IndexCastOp, arith::TruncIOp>(op)) {
      auto r = get(op->getOperand(0));
      return r ? fit(v.getType(), r->lo, r->hi) : Range{};
    }
    if (auto sel = dyn_cast<arith::SelectOp>(op)) {
      auto t = get(sel.getTrueValue()), f = get(sel.getFalseValue());
      if (!t || !f)
        return {};
      return fit(v.getType(), std::min(t->lo, f->lo), std::max(t->hi, f->hi));
    }
    if (isa<arith::MinSIOp>(op)) {
      auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
      if (a && b)
        return fit(v.getType(), std::min(a->lo, b->lo), std::min(a->hi, b->hi));
      if (b) {
        if (auto mx = op->getOperand(0).getDefiningOp<arith::MaxSIOp>()) {
          auto loBound = get(mx.getRhs());
          if (!loBound)
            loBound = get(mx.getLhs());
          if (loBound && loBound->lo <= b->lo)
            return fit(v.getType(), loBound->lo, b->hi);
        }
      }
      return {};
    }
    if (isa<arith::MaxSIOp>(op)) {
      auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
      if (!a || !b)
        return {};
      return fit(v.getType(), std::max(a->lo, b->lo), std::max(a->hi, b->hi));
    }
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
      auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
      if (!a || !b)
        return {};
      __int128 lo = a->lo, hi = a->hi, blo = b->lo, bhi = b->hi;
      if (isa<arith::AddIOp>(op))
        return fit(v.getType(), lo + blo, hi + bhi);
      if (isa<arith::SubIOp>(op))
        return fit(v.getType(), lo - bhi, hi - blo);
      auto extremes = std::minmax({lo * blo, lo * bhi, hi * blo, hi * bhi});
      return fit(v.getType(), extremes.first, extremes.second);
    }
    return {};
  }

public:
  explicit Bounds(func::FuncOp fn) : sym(SymbolBounds::of(fn)) {}
  Range get(Value v) {
    if (known.count(v))
      return known.lookup(v);
    known[v] = Range{};
    auto r = compute(v);
    known[v] = r;
    return r;
  }
};
} // namespace

bool canNarrowAddress(Operation &op, func::FuncOp fn) {
  if (op.getName().getStringRef() != "bangc_src.address")
    return false;
  auto segs = op.getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segs || segs.size() != 3 || segs[1] != segs[2])
    return false;
  unsigned n = segs[1];
  auto coords = op.getOperands().slice(1, n);
  auto strides = op.getOperands().slice(1 + n, n);
  Bounds bounds(fn);
  Interval sum{0, 0};
  auto i64Ty = IntegerType::get(op.getContext(), 64);
  for (auto [c, s] : llvm::zip(coords, strides)) {
    auto rc = bounds.get(c), rs = bounds.get(s);
    if (!rc || !rs)
      return false;
    __int128 cl = rc->lo, ch = rc->hi, sl = rs->lo, sh = rs->hi;
    auto extremes = std::minmax({cl * sl, cl * sh, ch * sl, ch * sh});
    auto term = fit(i64Ty, extremes.first, extremes.second);
    if (!term)
      return false;
    auto next = fit(i64Ty, __int128(sum.lo) + term->lo,
                    __int128(sum.hi) + term->hi);
    if (!next)
      return false;
    sum = *next;
  }
  return true;
}

namespace {
struct NarrowProvenAddresses
    : PassWrapper<NarrowProvenAddresses, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NarrowProvenAddresses)
  void runOnOperation() final {
    auto m = getOperation();
    auto funcs = m.getOps<func::FuncOp>();
    if (!llvm::hasSingleElement(funcs))
      return;
    auto fn = *funcs.begin();
    Builder b(&getContext());
    m.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "bangc_src.address" &&
          canNarrowAddress(*op, fn))
        op->setAttr("index_bits", b.getI32IntegerAttr(32));
    });
  }
};
} // namespace

void addressOptimization(compilerinfra::Unit &u) {
  u.contributions.push_back(
      {{"narrow-proven-addresses",
        "bangc@callable",
        [](OpPassManager &pm, const compilerinfra::TargetProfile &) {
          pm.addPass(std::make_unique<NarrowProvenAddresses>());
          return llvm::Error::success();
        },
        {},
        {},
        false},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
