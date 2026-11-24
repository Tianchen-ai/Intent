#include "../legalize/Rows.h"
#include "compilerinfra/Proofs.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <climits>
#include <optional>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;

namespace {
struct Interval {
  int64_t lo, hi;
};

std::optional<Interval> fitI32(__int128 lo, __int128 hi) {
  if (lo > hi || lo < INT32_MIN || hi > INT32_MAX)
    return std::nullopt;
  return Interval{static_cast<int64_t>(lo), static_cast<int64_t>(hi)};
}
} // namespace

bool canNarrowAddress(Operation &op, func::FuncOp fn,
                      const compilerinfra::IntegerRangeContext *existing) {
  if (op.getName().getStringRef() != "bangc_src.address")
    return false;
  auto segs = op.getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segs || segs.size() != 3 || segs[1] < 0 || segs[1] != segs[2])
    return false;
  unsigned n = static_cast<unsigned>(segs[1]);
  if (n == 0)
    return true;

  auto coords = op.getOperands().slice(1, n);
  auto strides = op.getOperands().slice(1 + n, n);
  SmallVector<Value, 8> addrInputs;
  addrInputs.reserve(2 * n);
  addrInputs.append(coords.begin(), coords.end());
  addrInputs.append(strides.begin(), strides.end());

  std::optional<compilerinfra::IntegerRangeContext> localCtx;
  if (!existing)
    localCtx.emplace(fn.getOperation());
  const auto &ranges = existing ? *existing : *localCtx;
  if (!ranges.valid())
    return false;

  compilerinfra::AddressProofOptions opts;
  opts.targetBitwidth = 32;
  opts.fallbackBitwidth = 64;
  opts.isSigned = true;
  if (!compilerinfra::proveNarrowAddress(fn.getOperation(), addrInputs, opts,
                                         &ranges)
           .proven)
    return false;

  Interval sum{0, 0};
  for (auto [c, s] : llvm::zip(coords, strides)) {
    auto rc = ranges.query(c), rs = ranges.query(s);
    if (!rc || !rs || rc->smin().getBitWidth() > 64 ||
        rs->smin().getBitWidth() > 64)
      return false;
    auto cFit = fitI32(rc->smin().getSExtValue(), rc->smax().getSExtValue());
    auto sFit = fitI32(rs->smin().getSExtValue(), rs->smax().getSExtValue());
    if (!cFit || !sFit)
      return false;
    __int128 cl = cFit->lo, ch = cFit->hi, sl = sFit->lo, sh = sFit->hi;
    auto extremes = std::minmax({cl * sl, cl * sh, ch * sl, ch * sh});
    auto term = fitI32(extremes.first, extremes.second);
    if (!term)
      return false;
    auto next = fitI32(static_cast<__int128>(sum.lo) + term->lo,
                       static_cast<__int128>(sum.hi) + term->hi);
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
    compilerinfra::IntegerRangeContext rangeCtx(fn.getOperation());
    Builder b(&getContext());
    m.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "bangc_src.address" &&
          canNarrowAddress(*op, fn, &rangeCtx))
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
