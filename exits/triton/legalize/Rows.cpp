#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

#include "intentdsl/levels/execution/block/ir/RowProgram.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
static StringRef binding(StringRef op) {
  return llvm::StringSwitch<StringRef>(op)
      .Case("triton_src.summary_reduce", "moments-reduction")
      .Case("triton_src.summary_merge", "moments-merge")
      .Case("triton_src.scan_sum", "inclusive-scan")
      .Case("triton_src.reduce", "native-row-reduce")
      .Default("");
}
void lowerRows(func::FuncOp fn) {
  fn.walk<WalkOrder::PostOrder>([](Operation *op) {
    auto code = op->getName().getStringRef();
    auto implementation = binding(code);
    if (!implementation.empty()) {
      Builder b(op->getContext());
      op->setAttr("ci.implementation",
                  b.getStringAttr(("exits/triton::" + implementation).str()));
    }
    if (code == "triton_src.offsets")
      op->setAttr("index_bits",
                  Builder(op->getContext()).getI32IntegerAttr(64));
    if (op->getName().getDialectNamespace() == "triton_src")
      intentdsl::block::expandRowOperation(*op, "exits/triton", true);
  });
}
LogicalResult verifyDynamicRows(func::FuncOp fn) {
  return intentdsl::block::verifyRowProgram(fn, "triton_src");
}
} // namespace CI_UNIT_NAMESPACE
