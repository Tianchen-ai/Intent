#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include <limits>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
struct Interval {
  int64_t lo, hi;
};
using Range = std::optional<Interval>;
Range i32(__int128 lo, __int128 hi) {
  if (lo < INT32_MIN || hi > INT32_MAX || lo > hi)
    return {};
  return Interval{int64_t(lo), int64_t(hi)};
}
Range fitSigned(__int128 lo, __int128 hi, Type t) {
  Type elem = getElementTypeOrSelf(t);
  unsigned bits = 0;
  if (elem.isIndex())
    bits = 64;
  else if (auto it = dyn_cast<IntegerType>(elem); it && it.isSignless())
    bits = it.getWidth();
  if (bits == 0 || bits > 64 || lo > hi)
    return {};
  __int128 minVal = INT32_MIN, maxVal = INT32_MAX;
  if (bits < 32) {
    minVal = -(static_cast<__int128>(1) << (bits - 1));
    maxVal = (static_cast<__int128>(1) << (bits - 1)) - 1;
  }
  if (lo < minVal || hi > maxVal)
    return {};
  return Interval{int64_t(lo), int64_t(hi)};
}
// Only static operations for which all lanes have a proved signed range
// without wraparound in their declared integer width.
class Bounds {
  llvm::DenseMap<Value, Range> known;
  Range compute(Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
      if (!loop || loop.getInductionVar() != v)
        return {};
      auto a = get(loop.getLowerBound()), z = get(loop.getUpperBound()),
           s = get(loop.getStep());
      if (!a || !z || !s || s->lo <= 0 || a->lo >= z->hi)
        return {};
      return fitSigned(a->lo, static_cast<__int128>(z->hi) - 1, v.getType());
    }
    auto *op = v.getDefiningOp();
    if (!op)
      return {};
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
        return fitSigned(attr.getValue().getSExtValue(),
                         attr.getValue().getSExtValue(), v.getType());
      if (auto dense = dyn_cast<DenseIntElementsAttr>(c.getValue());
          dense && dense.isSplat()) {
        auto val = dense.getSplatValue<APInt>().getSExtValue();
        return fitSigned(val, val, v.getType());
      }
      return {};
    }
    auto n = op->getName().getStringRef();
    if (n == "triton_src.tile_axis")
      return fitSigned(0, cast<RankedTensorType>(v.getType()).getDimSize(0) - 1,
                       v.getType());
    if (n == "triton_src.tile_broadcast" ||
        isa<arith::IndexCastOp, arith::ExtSIOp>(op)) {
      auto r = get(op->getOperand(0));
      return r ? fitSigned(r->lo, r->hi, v.getType()) : Range{};
    }
    if (!isa<arith::MulIOp, arith::AddIOp, arith::SubIOp>(op))
      return {};
    auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
    if (!a || !b)
      return {};
    __int128 al = a->lo, ah = a->hi, bl = b->lo, bh = b->hi;
    if (isa<arith::AddIOp>(op))
      return fitSigned(al + bl, ah + bh, v.getType());
    if (isa<arith::SubIOp>(op))
      return fitSigned(al - bh, ah - bl, v.getType());
    auto extremes = std::minmax({al * bl, al * bh, ah * bl, ah * bh});
    return fitSigned(extremes.first, extremes.second, v.getType());
  }

public:
  Range get(Value v) {
    auto it = known.find(v);
    if (it != known.end())
      return it->second;
    auto r = compute(v);
    known[v] = r;
    return r;
  }
};
} // namespace
bool canNarrowAddress(Operation &op) {
  auto strides = op.getAttrOfType<DenseI64ArrayAttr>("strides");
  auto indices = op.getOperands().drop_front();
  if (!strides || strides.size() != indices.size())
    return false;
  Bounds bounds;
  Interval sum{0, 0};
  for (auto [v, stride] : llvm::zip(indices, strides.asArrayRef())) {
    auto r = bounds.get(v);
    if (!r || stride < INT32_MIN || stride > INT32_MAX)
      return false;
    __int128 l = static_cast<__int128>(r->lo) * stride,
             h = static_cast<__int128>(r->hi) * stride;
    if (l > h)
      std::swap(l, h);
    auto term = i32(l, h),
         next = i32(static_cast<__int128>(sum.lo) + l,
                    static_cast<__int128>(sum.hi) + h);
    if (!term || !next)
      return false;
    sum = *next;
  }
  return true;
}
namespace {
struct Narrow : PassWrapper<Narrow, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Narrow)
  void runOnOperation() final {
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "triton_src.tile_address" &&
          canNarrowAddress(*op))
        op->setAttr("index_bits", Builder(&getContext()).getI32IntegerAttr(32));
    });
  }
};
} // namespace
void addressOptimization(compilerinfra::Unit &u) {
  u.contributions.push_back(
      {{"narrow-proven-addresses",
        "triton@callable",
        [](OpPassManager &pm, const compilerinfra::TargetProfile &) {
          pm.addPass(std::make_unique<Narrow>());
          return llvm::Error::success();
        },
        {},
        {},
        false},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
