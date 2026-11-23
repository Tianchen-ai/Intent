#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Normalized.h"
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace CI_UNIT_NAMESPACE {
inline mlir::LogicalResult memberAccesses(mlir::func::FuncOp fn, bool assign) {
  using namespace mlir;
  auto &space = fn.front().front().getRegion(0).front().front();
  auto cached = space.getAttrOfType<IntegerAttr>("cpu.score_cache");
  int panel = 0;
  if (cached && cached.getInt() >= 16 && cached.getInt() % 16 == 0) {
    auto shape = [&](unsigned i) {
      return fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.shape");
    };
    bool dense = true;
    for (unsigned i = 0; i < 4; ++i)
      dense &= fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.strides")[3] == 1;
    if (dense && shape(0)[2] % 4 == 0 && llvm::isPowerOf2_64(shape(0)[3]) &&
        llvm::isPowerOf2_64(shape(2)[3]) && shape(0)[3] <= 128 &&
        shape(2)[3] <= 128 &&
        space.getAttrOfType<IntegerAttr>("cpu.grain").getInt() % 4 == 0)
      panel = 4;
  }
  auto attr = IntegerAttr::get(IntegerType::get(fn.getContext(), 64), panel);
  if (assign && panel)
    space.setAttr("mojo_row.score_microtile", attr);
  if (panel ? space.getAttr("mojo_row.score_microtile") != attr
            : space.hasAttr("mojo_row.score_microtile"))
    return space.emitError("normalized row microtile requires disjoint "
                           "complete query rows and contiguous member vectors");
  auto result = fn.walk([&](Operation *op) {
    auto kind = op->getName().stripDialect();
    bool across = space.hasAttr("cpu.owner_batch") &&
                  !op->hasAttr("cpu.owner_common") && op->getParentOp() &&
                  isa<scf::ForOp>(op->getParentOp()) &&
                  op->getNumResults() == 1 &&
                  isa<FloatType>(op->getResult(0).getType()) &&
                  (kind == "exp" || kind == "truncf" || kind == "extf");
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
