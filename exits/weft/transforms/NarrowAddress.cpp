#include "compilerinfra/Assembly.h"
#include "compilerinfra/Proofs.h"
#include "mlir/IR/BuiltinAttributes.h"
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

bool canNarrowWeftAddressWithContext(
    Operation &op, Operation *scopeRoot,
    const compilerinfra::IntegerRangeContext &ranges) {
  auto strides = op.getAttrOfType<DenseI64ArrayAttr>("strides");
  auto indices = op.getOperands();
  if (!strides || strides.size() != indices.size())
    return false;
  if (indices.empty())
    return true;

  compilerinfra::AddressProofOptions opts;
  opts.targetBitwidth = 32;
  opts.isSigned = true;
  llvm::SmallVector<Value, 8> roots(indices.begin(), indices.end());
  auto proof = compilerinfra::proveNarrowAddress(scopeRoot, roots, opts, &ranges);
  if (!proof.proven)
    return false;

  Interval sum{0, 0};
  for (auto [v, stride] : llvm::zip(indices, strides.asArrayRef())) {
    if (stride < INT32_MIN || stride > INT32_MAX)
      return false;
    auto cr = ranges.query(v);
    if (!cr)
      return false;
    auto r = fitI32(cr->smin().getSExtValue(), cr->smax().getSExtValue());
    if (!r)
      return false;
    __int128 l = __int128(r->lo) * stride, h = __int128(r->hi) * stride;
    if (l > h)
      std::swap(l, h);
    auto term = fitI32(l, h),
         next = fitI32(__int128(sum.lo) + l, __int128(sum.hi) + h);
    if (!term || !next)
      return false;
    sum = *next;
  }
  return true;
}

Operation *findWeftAddressScope(Operation &op) {
  for (Operation *p = op.getParentOp(); p; p = p->getParentOp())
    if (p->getName().getStringRef() == "weft_host.host_parallel_tiles")
      return p;
  return op.getParentOp();
}
} // namespace

bool canNarrowWeftAddress(Operation &op) {
  Operation * outerScope = findWeftAddressScope(op);
  if (!outerScope)
    return false;
  Operation *solverRoot = outerScope->getParentOp() ? outerScope->getParentOp()
                                                    : outerScope;
  compilerinfra::IntegerRangeContext ranges(solverRoot);
  return ranges.valid() &&
         canNarrowWeftAddressWithContext(op, outerScope, ranges);
}

namespace {
struct NarrowWeft : PassWrapper<NarrowWeft, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NarrowWeft)
  void runOnOperation() final {
    getOperation().walk([&](Operation *tiles) {
      if (tiles->getName().getStringRef() != "weft_host.host_parallel_tiles")
        return;
      Operation *solverRoot =
          tiles->getParentOp() ? tiles->getParentOp() : tiles;
      compilerinfra::IntegerRangeContext ranges(solverRoot);
      if (!ranges.valid())
        return;
      tiles->walk([&](Operation *op) {
        if (op->getName().getStringRef() == "weft_host.address" &&
            canNarrowWeftAddressWithContext(*op, tiles, ranges))
          op->setAttr("index_bits",
                      Builder(&getContext()).getI32IntegerAttr(32));
      });
    });
  }
};
} // namespace

void registerWeftAddressOptimization(compilerinfra::Unit &u) {
  u.contributions.push_back(
      {{"narrow-proven-addresses",
        "weft@realized",
        [](OpPassManager &pm, const compilerinfra::TargetProfile &) {
          pm.addPass(std::make_unique<NarrowWeft>());
          return llvm::Error::success();
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
