#pragma once
#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace CI_UNIT_NAMESPACE {
inline mlir::LogicalResult ownerVectors(mlir::func::FuncOp fn, bool assign,
                                        int64_t budget) {
  using namespace mlir;
  auto &space = fn.front().front();
  auto trAttr = space.getAttrOfType<StringAttr>("dsa.transfers");
  if (trAttr && trAttr.getValue() == "panel-fold") {
    auto zero = IntegerAttr::get(IntegerType::get(fn.getContext(), 64), 0);
    if (assign)
      space.setAttr("bangc.deferred_bytes", zero);
    return success(space.getAttr("bangc.deferred_bytes") == zero);
  }
  auto streamBytes = space.getAttrOfType<IntegerAttr>("dsa.stream_bytes");
  auto memberBytes = space.getAttrOfType<IntegerAttr>("dsa.member_bytes");
  auto packAttr = space.getAttrOfType<IntegerAttr>("dsa.owner_pack");
  int64_t pack = packAttr ? packAttr.getInt() : 1;
  int64_t available = budget - (streamBytes ? streamBytes.getInt() : 0) -
                      (memberBytes ? memberBytes.getInt() : 0);
  llvm::SmallDenseSet<Operation *> deferred;
  int64_t scratch = 0;
  for (auto loop : space.getRegion(0).front().getOps<scf::ForOp>()) {
    for (auto st : loop.getBody()->getOps<memref::StoreOp>()) {
      auto transfer = st->getAttrOfType<DenseI64ArrayAttr>("dsa.stream");
      auto *sum = st.getValue().getDefiningOp();
      if (!transfer || !sum || sum->getName().stripDialect() != "lane_sum" ||
          !sum->hasOneUse())
        continue;
      auto *product = sum->getOperand(0).getDefiningOp();
      if (!product || product->getName().getStringRef() != "arith.mulf" ||
          !product->hasOneUse())
        continue;
      auto k =
          intentdsl::kir::memberExtent(fn, sum->getOperand(0), "block.shape");
      int64_t totalRows = transfer[2] * pack;
      int64_t chunkRows =
          pack > 1 ? std::min<int64_t>(transfer[2], 128) * pack : totalRows;
      int64_t defBytes = 0, workBytes = 0, size = 0;
      if (k != 32 || totalRows % 32 || totalRows > 2048 ||
          __builtin_mul_overflow(totalRows, 4 * k, &defBytes) ||
          __builtin_mul_overflow(chunkRows, 4 * k, &workBytes) ||
          __builtin_add_overflow(defBytes, workBytes, &size) ||
          size > available)
        continue;
      deferred.insert(st);
      deferred.insert(sum);
      deferred.insert(product);
      scratch = std::max(scratch, size);
      break;
    }
  }
  auto needed =
      IntegerAttr::get(IntegerType::get(fn.getContext(), 64), scratch);
  if (assign)
    space.setAttr("bangc.deferred_bytes", needed);
  else if (space.getAttr("bangc.deferred_bytes") != needed)
    return space.emitError("deferred projection exceeds local footprint");
  auto walk = fn.walk([&](Operation *op) {
    auto name = op->getName().stripDialect();
    StringRef mode;
    if (name == "lane_sum" &&
        intentdsl::kir::memberExtent(fn, op->getOperand(0), "block.shape") >
            32 &&
        available - scratch >= 128)
      mode = "pool-members";
    if (name == "lane_sum" && pack == 1 && op->getOperand(0).hasOneUse() &&
        op->getOperand(0).getDefiningOp() &&
        op->getOperand(0).getDefiningOp()->getName().getStringRef() ==
            "arith.mulf" &&
        intentdsl::kir::memberExtent(fn, op->getOperand(0), "block.shape") ==
            32)
      mode = "reduce32-reuse-product";
    if (name == "slice_read" && op->hasAttr("dsa.stream"))
      mode = "stream-view";
    if (name == "lane_splat" && op->getResult(0).hasOneUse() &&
        (*op->getResult(0).getUsers().begin())->getName().getStringRef() ==
            "arith.mulf" &&
        intentdsl::kir::memberExtent(fn, op->getResult(0), "block.shape") %
                32 ==
            0 &&
        (pack == 1 || !intentdsl::dsa::ownerValueDependsOn(
                          op->getOperand(0),
                          space.getRegion(0).front().getArguments().back())))
      mode = "broadcast-multiplier";
    if (name == "extf" && isa<RankedTensorType>(op->getOperand(0).getType()) &&
        cast<RankedTensorType>(op->getOperand(0).getType())
            .getElementType()
            .isBF16() &&
        intentdsl::kir::memberExtent(fn, op->getOperand(0), "block.shape") %
                32 ==
            0)
      mode = "widen-bits";
    if ((name == "extf" || name == "truncf") &&
        isa<RankedTensorType>(op->getOperand(0).getType()) &&
        intentdsl::kir::memberExtent(fn, op->getOperand(0), "block.shape") %
                32 ==
            0) {
      auto from =
          cast<RankedTensorType>(op->getOperand(0).getType()).getElementType();
      auto to =
          cast<RankedTensorType>(op->getResult(0).getType()).getElementType();
      if (from.isF16() && to.isF32())
        mode = "widen-half";
      if (from.isF32() && to.isF16())
        mode = "narrow-half";
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (loop.getNumResults() == 1 &&
          isa<RankedTensorType>(loop.getResult(0).getType()))
        mode = "single-carry";
    }
    if (deferred.contains(op))
      mode = "deferred-row-reduction";
    if (mode.empty()) {
      if (!op->hasAttr("bangc.vector_impl"))
        return WalkResult::advance();
    } else {
      auto attr = StringAttr::get(fn.getContext(), mode);
      if (assign)
        op->setAttr("bangc.vector_impl", attr);
      if (op->getAttr("bangc.vector_impl") == attr)
        return WalkResult::advance();
    }
    op->emitError("invalid native implementation for this member extent, "
                  "source or lifetime");
    return WalkResult::interrupt();
  });
  return success(!walk.wasInterrupted());
}
} // namespace CI_UNIT_NAMESPACE
