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

Range clampExtent(int64_t extent) {
  if (extent <= 1)
    return Interval{0, 0};
  return i32(0, extent - 1);
}

class Bounds {
  llvm::DenseMap<Value, Range> known;

  Range compute(Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      auto *parent = arg.getOwner()->getParentOp();
      if (!parent ||
          parent->getName().getStringRef() != "weft_host.host_parallel_tiles")
        return {};
      auto ext = parent->getAttrOfType<DenseI64ArrayAttr>("extents");
      auto tile = parent->getAttrOfType<DenseI64ArrayAttr>("tile");
      if (!ext || ext.size() < 4 || !tile || tile.size() < 2)
        return {};
      int64_t M = ext[0], N = ext[1], G = ext[3];
      int64_t mr = tile[0], nr = tile[1];
      int64_t mult =
          std::max<int64_t>(1, parent->getAttrOfType<IntegerAttr>("multiplicity").getInt());
      int64_t H =
          std::max<int64_t>(1, parent->getAttrOfType<IntegerAttr>("head_extent").getInt());
      int64_t chunks =
          std::max<int64_t>(1, parent->getAttrOfType<IntegerAttr>("chunk_extent").getInt());
      bool routed = parent->getAttrOfType<BoolAttr>("routed").getValue();

      // Block args: (%g, %c, %r, %src, %dest, %batch, %chunk, %head, %R, %C)
      switch (arg.getArgNumber()) {
      case 0: // g
        return clampExtent(G);
      case 1: // c
        return clampExtent(N);
      case 2: // r
        return clampExtent(M);
      case 3: // src
        return clampExtent(M);
      case 4: // dest
        return clampExtent(routed ? M * mult : M);
      case 5: // batch
        return clampExtent(std::max<int64_t>(1, G / (H * chunks)));
      case 6: // chunk
        return clampExtent(chunks);
      case 7: // head
        return clampExtent(H);
      case 8: // R
        return i32(0, std::max<int64_t>(0, mr));
      case 9: // C
        return i32(0, std::max<int64_t>(0, nr));
      default:
        return {};
      }
    }

    auto *op = v.getDefiningOp();
    if (!op)
      return {};
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
        return i32(attr.getInt(), attr.getInt());
      return {};
    }
    if (isa<arith::MinSIOp>(op)) {
      auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
      if (!a || !b)
        return {};
      return i32(std::min(a->lo, b->lo), std::min(a->hi, b->hi));
    }
    if (isa<arith::DivSIOp>(op)) {
      auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
      if (!a || !b || a->lo < 0 || b->lo <= 0)
        return {};
      return i32(a->lo / b->hi, a->hi / b->lo);
    }
    if (isa<arith::RemSIOp>(op)) {
      auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
      if (!a || !b || a->lo < 0 || b->lo <= 0)
        return {};
      return i32(0, std::min(a->hi, b->hi - 1));
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

bool canNarrowWeftAddress(Operation &op) {
  auto strides = op.getAttrOfType<DenseI64ArrayAttr>("strides");
  auto indices = op.getOperands();
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
    auto term = i32(l, h),
         next = i32(__int128(sum.lo) + l, __int128(sum.hi) + h);
    if (!term || !next)
      return false;
    sum = *next;
  }
  return true;
}

namespace {
struct NarrowWeft : PassWrapper<NarrowWeft, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NarrowWeft)
  void runOnOperation() final {
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "weft_host.address" &&
          canNarrowWeftAddress(*op))
        op->setAttr("index_bits", Builder(&getContext()).getI32IntegerAttr(32));
    });
  }
};
} // namespace

void registerWeftAddressOptimization(compilerinfra::Unit &u) {
  u.contributions.push_back(
      {{"narrow-proven-addresses",
        "weft@callable",
        [](OpPassManager &pm, const compilerinfra::TargetProfile &) {
          pm.addPass(std::make_unique<NarrowWeft>());
          return llvm::Error::success();
        },
        {},
        {},
        false},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
