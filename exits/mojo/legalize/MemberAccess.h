#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace CI_UNIT_NAMESPACE {
inline mlir::LogicalResult memberAccesses(mlir::func::FuncOp fn, bool assign) {
  using namespace mlir;
  auto &space = fn.front().front().getRegion(0).front().front();
  if (space.hasAttr("mojo_row.score_microtile"))
    return space.emitError("unexpected legacy score_microtile attribute");
  auto result = fn.walk([&](Operation *op) {
    auto kind = op->getName().stripDialect();
    bool across = space.hasAttr("cpu.owner_batch") &&
                  !op->hasAttr("cpu.owner_common") && op->getParentOp() &&
                  isa<scf::ForOp>(op->getParentOp()) &&
                  op->getNumResults() == 1 &&
                  isa<FloatType>(op->getResult(0).getType()) &&
                  (kind == "exp" || kind == "exp2");
    if (assign && across)
      op->setAttr("mojo_row.pack_scalars", UnitAttr::get(op->getContext()));
    if (op->hasAttr("mojo_row.pack_scalars") != across)
      return WalkResult::interrupt();
    bool vector = false;
    if (kind == "slice_read" || kind == "slice_write") {
      auto base =
          cast<BlockArgument>(op->getOperand(kind == "slice_write" ? 1 : 0));
      auto ax = op->getAttrOfType<IntegerAttr>("axis").getInt();
      auto strides = fn.getArgAttrOfType<DenseI64ArrayAttr>(base.getArgNumber(),
                                                            "cpu.strides");
      auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(base.getArgNumber(),
                                                          "cpu.shape");
      vector = strides && shape && strides[ax] == 1 &&
               llvm::isPowerOf2_64(shape[ax]);
    }
    if (assign && vector)
      op->setAttr("mojo_row.contiguous_simd", UnitAttr::get(op->getContext()));
    return vector == op->hasAttr("mojo_row.contiguous_simd")
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  return result.wasInterrupted()
             ? fn.emitError(
                   "SIMD load/store requires a full contiguous member slice")
             : success();
}
} // namespace CI_UNIT_NAMESPACE
