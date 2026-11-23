#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
namespace CI_UNIT_NAMESPACE {
struct MemberStore {
  mlir::Value vector, scale;
};
inline std::optional<MemberStore> scaledMemberStore(mlir::Operation *op) {
  using namespace mlir;
  if (op->getName().stripDialect() != "slice_write")
    return {};
  auto narrow = op->getOperand(0).getDefiningOp<arith::TruncFOp>();
  if (!narrow || !narrow.getResult().hasOneUse())
    return {};
  auto product = narrow.getIn().getDefiningOp<arith::MulFOp>();
  if (!product || !product.getResult().hasOneUse())
    return {};
  for (unsigned j : {0u, 1u}) {
    auto *def = product->getOperand(j).getDefiningOp();
    if (def && def->getName().stripDialect() == "lane_splat")
      return MemberStore{product->getOperand(1 - j), def->getOperand(0)};
  }
  return {};
}
inline mlir::LogicalResult bindMemberStores(mlir::Operation &task, bool rvv,
                                            bool assign) {
  auto result = task.walk([&](mlir::Operation *op) {
    bool use = rvv && scaledMemberStore(op).has_value();
    auto attr = mlir::StringAttr::get(op->getContext(), "scale-narrow-store");
    if (assign && use)
      op->setAttr("weft_host.member_store", attr);
    if (use ? op->getAttr("weft_host.member_store") != attr
            : op->hasAttr("weft_host.member_store")) {
      op->emitError(
          "lane publication requires scaled and narrowed member values");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return mlir::success(!result.wasInterrupted());
}
} // namespace CI_UNIT_NAMESPACE
