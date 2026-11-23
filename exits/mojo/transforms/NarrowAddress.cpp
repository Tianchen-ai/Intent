#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
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
class Bounds {
  llvm::DenseMap<Value, Range> known;
  Range compute(Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      auto *parent = arg.getOwner()->getParentOp();
      if (!parent)
        return {};
      auto n = parent->getName().getStringRef();
      if (n == "mojo_row.for_range" && arg.getArgNumber() == 0) {
        auto a = get(parent->getOperand(0)), z = get(parent->getOperand(1)),
             s = get(parent->getOperand(2));
        if (!a || !z || !s || s->lo <= 0)
          return {};
        if (a->lo >= z->hi)
          return i32(0, 0);
        return i32(a->lo, z->hi - 1);
      }
      if (n == "mojo_row.while_lt" && arg.getArgNumber() == 0) {
        auto a = get(parent->getOperand(0)), z = get(parent->getOperand(1)),
             s = get(parent->getOperand(2));
        if (!a || !z || !s || s->lo <= 0)
          return {};
        if (a->lo >= z->hi)
          return i32(a->lo, a->hi);
        return i32(a->lo, z->hi - 1);
      }
      return {};
    }
    auto *op = v.getDefiningOp();
    if (!op)
      return {};
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
        return i32(attr.getInt(), attr.getInt());
      return {};
    }
    auto n = op->getName().getStringRef();
    if (n == "mojo_row.as_int")
      return get(op->getOperand(0));
    if (isa<arith::MinSIOp>(op)) {
      auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
      if (!a || !b)
        return {};
      return i32(std::min(a->lo, b->lo), std::min(a->hi, b->hi));
    }
    if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op))
      return {};
    auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
    if (!a || !b)
      return {};
    if (isa<arith::AddIOp>(op))
      return i32(__int128(a->lo) + b->lo, __int128(a->hi) + b->hi);
    if (isa<arith::SubIOp>(op))
      return i32(__int128(a->lo) - b->hi, __int128(a->hi) - b->lo);
    auto extremes =
        std::minmax({__int128(a->lo) * b->lo, __int128(a->lo) * b->hi,
                     __int128(a->hi) * b->lo, __int128(a->hi) * b->hi});
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

bool canNarrowMojoAddress(Operation &op) {
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
    __int128 l = __int128(r->lo) * stride, h = __int128(r->hi) * stride;
    if (l > h)
      std::swap(l, h);
    auto term = i32(l, h), next = i32(__int128(sum.lo) + l, __int128(sum.hi) + h);
    if (!term || !next)
      return false;
    sum = *next;
  }
  return true;
}

namespace {
struct NarrowMojo : PassWrapper<NarrowMojo, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NarrowMojo)
  void runOnOperation() final {
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "mojo_row.address" &&
          canNarrowMojoAddress(*op))
        op->setAttr("index_bits", Builder(&getContext()).getI32IntegerAttr(32));
    });
  }
};
} // namespace

void registerMojoAddressOptimization(compilerinfra::Unit &u) {
  u.contributions.push_back(
      {{"narrow-proven-addresses",
        "mojo@callable",
        [](OpPassManager &pm, const compilerinfra::TargetProfile &) {
          pm.addPass(std::make_unique<NarrowMojo>());
          return llvm::Error::success();
        },
        {},
        {},
        false},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
