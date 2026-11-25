#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "intentdsl/levels/execution/block/ir/RowProgram.h"
#include <climits>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
bool canNarrowRowCoordinates(func::FuncOp fn,
                             const compilerinfra::TargetProfile &profile) {
  const auto *facts = profile.services.get<intentdsl::CallFacts>();
  const auto *caps = profile.services.get<intentdsl::RowCapacity>();
  if (!facts || !caps)
    return false;
  auto n = caps->width(fn);
  if (!n || *n < 0 || *n > INT32_MAX - 8192)
    return false;
  if (auto bounds = fn->getAttrOfType<DenseI64ArrayAttr>("kir.iteration")) {
    auto owner = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
    int64_t m = 1;
    for (int i = 0; i < bounds.size(); i += 4) {
      if (owner ? i / 4 != owner.getInt() : i + 4 == bounds.size())
        continue;
      auto ax = bounds.asArrayRef().slice(i, 4);
      int64_t ext = ax[3];
      if (ax[0] >= 0) {
        if (size_t(ax[0]) >= facts->arguments.size() || ax[2] <= 0)
          return false;
        auto *obj = facts->arguments[ax[0]].getAsObject();
        auto *sh = obj ? obj->getArray("shape") : nullptr;
        if (!sh || ax[1] < 0 || size_t(ax[1]) >= sh->size())
          return false;
        auto d = (*sh)[ax[1]].getAsInteger();
        if (!d || *d < 0)
          return false;
        ext = *d / ax[2];
      }
      if (ext < 0 || __builtin_mul_overflow(m, ext, &m) ||
          m > INT32_MAX - 1024)
        return false;
    }
    return true;
  }
  auto [m, cols] = facts->geometry(fn);
  return m >= 0 && m <= INT32_MAX - 1024 && cols >= 0 &&
         cols <= INT32_MAX - 8192;
}
} // namespace

static StringRef binding(StringRef op) {
  return llvm::StringSwitch<StringRef>(op)
      .Case("triton_src.summary_reduce", "moments-reduction")
      .Case("triton_src.summary_merge", "moments-merge")
      .Case("triton_src.scan_sum", "inclusive-scan")
      .Case("triton_src.reduce", "native-row-reduce")
      .Default("");
}
void lowerRows(func::FuncOp fn, const compilerinfra::TargetProfile &profile) {
  int32_t coordBits = canNarrowRowCoordinates(fn, profile) ? 32 : 64;
  Builder fb(fn.getContext());
  bool grid2d = fn->getAttr("block.grid") == intentdsl::block::grid(fb, true);
  auto ownerAxis = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
  fn.walk<WalkOrder::PostOrder>([&](Operation *op) {
    auto code = op->getName().getStringRef();
    auto implementation = binding(code);
    if (!implementation.empty()) {
      op->setAttr("ci.implementation",
                  fb.getStringAttr(("exits/triton::" + implementation).str()));
    }
    if (code == "triton_src.pid")
      op->setAttr("grid_2d", fb.getBoolAttr(grid2d));
    if (code == "triton_src.offsets") {
      op->setAttr("index_bits", fb.getI32IntegerAttr(coordBits));
      if (ownerAxis)
        op->setAttr("triton_src.owner_axis",
                    fb.getI64IntegerAttr(ownerAxis.getInt()));
    }
    if (op->getName().getDialectNamespace() == "triton_src")
      intentdsl::block::expandRowOperation(*op, "exits/triton", true);
  });
}
LogicalResult verifyDynamicRows(func::FuncOp fn,
                                const compilerinfra::TargetProfile &profile) {
  int32_t expectedBits = canNarrowRowCoordinates(fn, profile) ? 32 : 64;
  Builder b(fn.getContext());
  auto expectedAttr = b.getI32IntegerAttr(expectedBits);
  auto expectedGrid2d =
      b.getBoolAttr(fn->getAttr("block.grid") == intentdsl::block::grid(b, true));
  auto ownerAttr = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
  Attribute expectedOwner =
      ownerAttr ? Attribute(b.getI64IntegerAttr(ownerAttr.getInt()))
                : Attribute();
  if (fn.walk([&](Operation *op) {
          auto code = op->getName().getStringRef();
          if (code == "triton_src.pid" &&
              op->getAttr("grid_2d") != expectedGrid2d) {
            op->emitError("triton_src.pid grid_2d disagrees with block.grid");
            return WalkResult::interrupt();
          }
          if (code == "triton_src.offsets" &&
              (op->getAttr("index_bits") != expectedAttr ||
               op->getAttr("triton_src.owner_axis") != expectedOwner)) {
            op->emitError("triton_src.offsets attributes disagree with proven layout");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        }).wasInterrupted())
    return failure();
  return intentdsl::block::verifyRowProgram(fn, "triton_src");
}
} // namespace CI_UNIT_NAMESPACE
