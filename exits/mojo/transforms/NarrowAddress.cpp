#include "compilerinfra/Assembly.h"
#include "compilerinfra/Proofs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

Range fitI32(__int128 lo, __int128 hi) {
  if (lo < INT32_MIN || hi > INT32_MAX || lo > hi)
    return {};
  return Interval{int64_t(lo), int64_t(hi)};
}

Range evaluateMojoAddressRange(Operation &op, Operation *scopeRoot,
                               const compilerinfra::IntegerRangeContext &ranges,
                               llvm::DenseMap<Value, Range> &memo);

// Evaluates Mojo terminal index expressions by delegating standard scalar
// arithmetic subtrees to compilerinfra::proveNarrowAddress + IntegerRangeContext
// while bridging mojo_row loop/coord wrappers.
Range evaluateMojoValueRange(Value v, Operation *scopeRoot,
                             const compilerinfra::IntegerRangeContext &ranges,
                             llvm::DenseMap<Value, Range> &memo) {
  if (auto it = memo.find(v); it != memo.end())
    return it->second;
  memo[v] = std::nullopt;
  Range res;

  compilerinfra::AddressProofOptions opts;
  opts.targetBitwidth = 32;
  opts.isSigned = true;
  auto direct = compilerinfra::proveNarrowAddress(scopeRoot, {v}, opts, &ranges);
  if (direct.proven) {
    if (auto cr = ranges.query(v)) {
      res = fitI32(cr->smin().getSExtValue(), cr->smax().getSExtValue());
      if (res) {
        memo[v] = res;
        return res;
      }
    }
  }

  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *parent = arg.getOwner()->getParentOp();
    if (parent && arg.getArgNumber() == 0) {
      StringRef n = parent->getName().getStringRef();
      if (n == "mojo_row.for_range") {
        auto a = evaluateMojoValueRange(parent->getOperand(0), scopeRoot,
                                        ranges, memo);
        auto z = evaluateMojoValueRange(parent->getOperand(1), scopeRoot,
                                        ranges, memo);
        auto s = evaluateMojoValueRange(parent->getOperand(2), scopeRoot,
                                        ranges, memo);
        if (a && z && s && s->lo > 0)
          res = a->lo >= z->hi ? fitI32(0, 0) : fitI32(a->lo, z->hi - 1);
      } else if (n == "mojo_row.while_lt") {
        auto a = evaluateMojoValueRange(parent->getOperand(0), scopeRoot,
                                        ranges, memo);
        auto z = evaluateMojoValueRange(parent->getOperand(1), scopeRoot,
                                        ranges, memo);
        auto s = evaluateMojoValueRange(parent->getOperand(2), scopeRoot,
                                        ranges, memo);
        if (a && z && s && s->lo > 0)
          res = a->lo >= z->hi ? fitI32(a->lo, a->hi) : fitI32(a->lo, z->hi - 1);
      }
    }
    memo[v] = res;
    return res;
  }

  Operation *op = v.getDefiningOp();
  if (!op)
    return {};
  StringRef n = op->getName().getStringRef();
  if (n == "mojo_row.as_int") {
    res = evaluateMojoValueRange(op->getOperand(0), scopeRoot, ranges, memo);
  } else if (n == "mojo_row.address") {
    res = evaluateMojoAddressRange(*op, scopeRoot, ranges, memo);
  } else if (n == "mojo_row.owner_coord") {
    if (auto cr = ranges.query(v))
      res = fitI32(cr->smin().getSExtValue(), cr->smax().getSExtValue());
  } else if (isa<arith::MinSIOp>(op)) {
    auto a = evaluateMojoValueRange(op->getOperand(0), scopeRoot, ranges, memo);
    auto b = evaluateMojoValueRange(op->getOperand(1), scopeRoot, ranges, memo);
    if (a && b)
      res = fitI32(std::min(a->lo, b->lo), std::min(a->hi, b->hi));
  } else if (isa<arith::FloorDivSIOp, arith::DivSIOp>(op)) {
    auto a = evaluateMojoValueRange(op->getOperand(0), scopeRoot, ranges, memo);
    auto b = evaluateMojoValueRange(op->getOperand(1), scopeRoot, ranges, memo);
    if (a && b && b->lo > 0 && b->lo == b->hi && a->lo >= 0)
      res = fitI32(a->lo / b->lo, a->hi / b->lo);
  } else if (isa<arith::RemSIOp>(op)) {
    auto a = evaluateMojoValueRange(op->getOperand(0), scopeRoot, ranges, memo);
    auto b = evaluateMojoValueRange(op->getOperand(1), scopeRoot, ranges, memo);
    if (a && b && b->lo > 0 && b->lo == b->hi && a->lo >= 0)
      res = fitI32(0, std::min(a->hi, b->lo - 1));
  } else if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
    auto a = evaluateMojoValueRange(op->getOperand(0), scopeRoot, ranges, memo);
    auto b = evaluateMojoValueRange(op->getOperand(1), scopeRoot, ranges, memo);
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

Range evaluateMojoAddressRange(Operation &op, Operation *scopeRoot,
                               const compilerinfra::IntegerRangeContext &ranges,
                               llvm::DenseMap<Value, Range> &memo) {
  auto strides = op.getAttrOfType<DenseI64ArrayAttr>("strides");
  auto indices = op.getOperands().drop_front();
  if (!strides || strides.size() != indices.size())
    return {};
  Interval sum{0, 0};
  for (auto [v, stride] : llvm::zip(indices, strides.asArrayRef())) {
    if (stride < INT32_MIN || stride > INT32_MAX)
      return {};
    auto r = evaluateMojoValueRange(v, scopeRoot, ranges, memo);
    if (!r)
      return {};
    __int128 l = __int128(r->lo) * stride, h = __int128(r->hi) * stride;
    if (l > h)
      std::swap(l, h);
    auto term = fitI32(l, h),
         next = fitI32(__int128(sum.lo) + l, __int128(sum.hi) + h);
    if (!term || !next)
      return {};
    sum = *next;
  }
  return sum;
}
} // namespace

bool canNarrowMojoAddress(Operation &op) {
  Operation *scope = op.getParentOfType<func::FuncOp>();
  if (!scope)
    scope = op.getParentOp();
  compilerinfra::IntegerRangeContext ranges(scope);
  if (!ranges.valid())
    return false;
  llvm::DenseMap<Value, Range> memo;
  return evaluateMojoAddressRange(op, scope, ranges, memo).has_value();
}

namespace {
struct NarrowMojo : PassWrapper<NarrowMojo, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NarrowMojo)
  void runOnOperation() final {
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      compilerinfra::IntegerRangeContext ranges(fn);
      if (!ranges.valid())
        continue;
      llvm::DenseMap<Value, Range> memo;
      fn.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "mojo_row.address" &&
            evaluateMojoAddressRange(*op, fn, ranges, memo).has_value())
          op->setAttr("index_bits",
                      Builder(&getContext()).getI32IntegerAttr(32));
      });
    }
  }
};
} // namespace

void registerMojoAddressOptimization(compilerinfra::Unit &u) {
  u.contributions.push_back(
      {{"narrow-proven-addresses",
        "mojo@realized",
        [](OpPassManager &pm, const compilerinfra::TargetProfile &) {
          pm.addPass(std::make_unique<NarrowMojo>());
          return llvm::Error::success();
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
