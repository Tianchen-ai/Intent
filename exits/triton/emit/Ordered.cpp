#include "intentdsl/levels/execution/block/ir/Owners.h"

using namespace mlir;
namespace CI_UNIT_NAMESPACE {
llvm::Expected<std::string> printColumns(func::FuncOp fn);
llvm::Error emitOrdered(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  if (!fn.front().front().hasAttr("block.column_prefixes"))
    return llvm::createStringError(
        "owner program missing terminal tile lowering");
  auto src = printColumns(fn);
  if (!src)
    return src.takeError();
  return intentdsl::block::emitOwnerArtifact(
      fn, *src, llvm::json::Object{{"num_warps", 4}, {"num_stages", 1}}, 2,
      out);
}
} // namespace CI_UNIT_NAMESPACE
