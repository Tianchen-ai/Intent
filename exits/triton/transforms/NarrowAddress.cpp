#include "compilerinfra/Assembly.h"
#include "compilerinfra/Proofs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

Range fitI32(__int128 lo, __int128 hi) {
  if (lo < INT32_MIN || hi > INT32_MAX || lo > hi)
    return {};
  return Interval{int64_t(lo), int64_t(hi)};
}

// Evaluates shaped tile lane expressions whose scalar roots are proven by
// compilerinfra::proveNarrowAddress + IntegerRangeContext.
Range evaluateTileRange(Value v, Operation *scopeRoot,
                        const compilerinfra::IntegerRangeContext &ranges,
                        llvm::DenseMap<Value, Range> &memo) {
  if (auto it = memo.find(v); it != memo.end())
    return it->second;
  memo[v] = std::nullopt;
  Range res;

  if (!isa<ShapedType>(v.getType())) {
    compilerinfra::AddressProofOptions opts;
    opts.targetBitwidth = 32;
    opts.isSigned = true;
    auto proof = compilerinfra::proveNarrowAddress(v, opts, &ranges);
    if (proof.proven) {
      if (auto cr = ranges.query(v)) {
        res = fitI32(cr->smin().getSExtValue(), cr->smax().getSExtValue());
      }
    }
    memo[v] = res;
    return res;
  }

  auto *op = v.getDefiningOp();
  if (!op)
    return {};
  Type elemTy = getElementTypeOrSelf(v.getType());
  if (auto it = dyn_cast<IntegerType>(elemTy);
      it && it.isSignless() && it.getWidth() > 0 && it.getWidth() <= 32) {
    unsigned w = it.getWidth();
    res = w == 32 ? Interval{INT32_MIN, INT32_MAX}
                  : Interval{-(int64_t(1) << (w - 1)),
                             (int64_t(1) << (w - 1)) - 1};
    memo[v] = res;
    return res;
  }
  if (auto c = dyn_cast<arith::ConstantOp>(op)) {
    if (auto dense = dyn_cast<DenseIntElementsAttr>(c.getValue());
        dense && dense.isSplat() &&
        dense.getElementType().getIntOrFloatBitWidth() <= 64) {
      auto val = dense.getSplatValue<APInt>().getSExtValue();
      res = fitI32(val, val);
    }
    memo[v] = res;
    return res;
  }
  auto n = op->getName().getStringRef();
  if (n == "triton_src.tile_axis") {
    res = fitI32(0, cast<RankedTensorType>(v.getType()).getDimSize(0) - 1);
  } else if (n == "triton_src.tile_broadcast" ||
             n == "triton_src.tile_prefix" ||
             n == "triton_src.drop_unit_owner" ||
             isa<arith::IndexCastOp, arith::ExtSIOp>(op)) {
    res = evaluateTileRange(op->getOperand(0), scopeRoot, ranges, memo);
  } else if (isa<arith::MulIOp, arith::AddIOp, arith::SubIOp>(op)) {
    auto a = evaluateTileRange(op->getOperand(0), scopeRoot, ranges, memo);
    auto b = evaluateTileRange(op->getOperand(1), scopeRoot, ranges, memo);
    if (a && b) {
      __int128 al = a->lo, ah = a->hi, bl = b->lo, bh = b->hi;
      if (isa<arith::AddIOp>(op))
        res = fitI32(al + bl, ah + bh);
      else if (isa<arith::SubIOp>(op))
        res = fitI32(al - bh, ah - bl);
      else {
        auto ex = std::minmax({al * bl, al * bh, ah * bl, ah * bh});
        res = fitI32(ex.first, ex.second);
      }
    }
  }
  memo[v] = res;
  return res;
}

bool canNarrowAddressWithContext(
    Operation &op, Operation *scopeRoot,
    const compilerinfra::IntegerRangeContext &ranges) {
  auto strides = op.getAttrOfType<DenseI64ArrayAttr>("strides");
  auto indices = op.getOperands().drop_front();
  if (!strides || strides.size() != indices.size())
    return false;
  llvm::DenseMap<Value, Range> memo;
  Interval sum{0, 0};
  for (auto [v, stride] : llvm::zip(indices, strides.asArrayRef())) {
    if (stride < INT32_MIN || stride > INT32_MAX)
      return false;
    auto r = evaluateTileRange(v, scopeRoot, ranges, memo);
    if (!r)
      return false;
    __int128 l = static_cast<__int128>(r->lo) * stride,
             h = static_cast<__int128>(r->hi) * stride;
    if (l > h)
      std::swap(l, h);
    auto term = fitI32(l, h),
         next = fitI32(static_cast<__int128>(sum.lo) + l,
                       static_cast<__int128>(sum.hi) + h);
    if (!term || !next)
      return false;
    sum = *next;
  }
  return true;
}
} // namespace

bool canNarrowAddress(Operation &op) {
  Operation *scope = op.getParentOfType<func::FuncOp>();
  if (!scope)
    scope = op.getParentOp();
  compilerinfra::IntegerRangeContext ranges(scope);
  return ranges.valid() && canNarrowAddressWithContext(op, scope, ranges);
}

namespace {
struct Narrow : PassWrapper<Narrow, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Narrow)
  void runOnOperation() final {
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      compilerinfra::IntegerRangeContext ranges(fn);
      if (!ranges.valid())
        continue;
      fn.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "triton_src.tile_address" &&
            canNarrowAddressWithContext(*op, fn, ranges))
          op->setAttr("index_bits",
                      Builder(&getContext()).getI32IntegerAttr(32));
      });
    }
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
        true},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
