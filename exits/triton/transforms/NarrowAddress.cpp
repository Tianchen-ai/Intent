#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
// Only static operations for which all lanes have a proved signed range.
// Unknown launch ids, ABI values and loaded indices keep full-width arithmetic.
class Bounds {
  llvm::DenseMap<Value, Range> known;
  Range compute(Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
      if (!loop || loop.getInductionVar() != v)
        return {};
      auto a = get(loop.getLowerBound()), z = get(loop.getUpperBound()),
           s = get(loop.getStep());
      if (!a || !z || !s || s->lo <= 0)
        return {};
      return i32(a->lo, z->hi - 1);
    }
    auto *op = v.getDefiningOp();
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
        return i32(attr.getInt(), attr.getInt());
      return {};
    }
    auto n = op->getName().getStringRef();
    if (n == "triton_src.tile_axis")
      return i32(0, cast<RankedTensorType>(v.getType()).getDimSize(0) - 1);
    if (n == "triton_src.tile_broadcast" ||
        isa<arith::IndexCastOp, arith::ExtSIOp>(op))
      return get(op->getOperand(0));
    if (!isa<arith::MulIOp, arith::AddIOp, arith::SubIOp>(op))
      return {};
    auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
    if (!a || !b)
      return {};
    if (isa<arith::AddIOp>(op))
      return i32(a->lo + b->lo, a->hi + b->hi);
    if (isa<arith::SubIOp>(op))
      return i32(a->lo - b->hi, a->hi - b->lo);
    auto extremes = std::minmax(
        {a->lo * b->lo, a->lo * b->hi, a->hi * b->lo, a->hi * b->hi});
    return i32(extremes.first, extremes.second);
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
    auto l = r->lo * stride, h = r->hi * stride;
    if (l > h)
      std::swap(l, h);
    auto term = i32(l, h), next = i32(sum.lo + l, sum.hi + h);
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
