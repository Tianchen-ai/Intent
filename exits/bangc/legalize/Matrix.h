#pragma once
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
namespace CI_UNIT_NAMESPACE {
inline mlir::LogicalResult bindMatrix(mlir::func::FuncOp fn, unsigned phase,
                                      bool write,
                                      const compilerinfra::TargetProfile &profile) {
  using namespace mlir;
  intentdsl::block::MatrixAccess p(fn);
  auto &setup = p.fold.getRegion(0).front().front();
  Builder b(fn.getContext());
  auto field = [&](Operation &op, StringRef key, Attribute a) {
    if (write)
      op.setAttr(key, a);
    return op.getAttr(key) == a;
  };
  if (setup.hasAttr("library_impl")) {
    if (!field(setup, "bangc.space", b.getStringAttr("external.cnnl")) ||
        !field(setup, "bangc.worker", b.getStringAttr("host-queue")))
      return setup.emitError("external matrix library binding mismatch");
    if (phase >= 1 &&
        (!field(p.dot, "bangc.matrix", b.getStringAttr("cnnl-matmul")) ||
         !field(setup, "bangc.writeback", b.getStringAttr("cnnl-direct"))))
      return setup.emitError("external matrix primitive binding mismatch");
    if (phase >= 2 &&
        !field(setup, "bangc.entry", b.getStringAttr("intent_enqueue")))
      return setup.emitError("missing matrix queue entry");
    return success();
  }
  if (!field(setup, "bangc.space", b.getStringAttr("private.nram")) ||
      !field(setup, "bangc.worker", b.getStringAttr("task-stride")))
    return setup.emitError("matrix task or local storage mismatch");
  if (phase >= 1) {
    auto a = p.dot.getOperand(0);
    auto type = cast<MemRefType>(a.getType()).getElementType();
    auto name = p.dot.getName().stripDialect();
    StringRef primitive = name == "q4k_dot"               ? "q4k-q8k"
                          : name == "scaled_contract"     ? "scaled-f32"
                          : name == "quant_contract"      ? "quant-format"
                          : type.isF16() || type.isBF16() ? "wram-conv"
                          : type.isInteger(8)             ? "exact-i8-conv"
                                                          : "f32-outer";
    if (!field(p.dot, "bangc.matrix", b.getStringAttr(primitive)))
      return p.dot.emitError("incorrect matrix microkernel");
    if ((primitive == "wram-conv" || primitive == "exact-i8-conv") &&
        p.tile[1] != 64)
      return p.dot.emitError(
          "native filter copy implements one 64-channel weight bank");
    bool allowFtz =
        compilerinfra::allowsSubnormalFlush(&p.dot, profile.numerical);
    if (type.isBF16() && !allowFtz &&
        !field(p.dot, "bangc.bf16_arithmetic",
               b.getStringAttr("checked-conv-ieee-fallback")))
      return p.dot.emitError(
          "BF16 dot requires an IEEE path outside the native exponent window");
    if (allowFtz && p.dot.hasAttr("bangc.bf16_arithmetic"))
      return p.dot.emitError(
          "BF16 subnormal FTZ contraction must not attach strict fallback");
    if (primitive == "exact-i8-conv" && p.tile[2] > 1024)
      return p.dot.emitError(
          "signed byte partial dot exceeds exact binary32 integer range");
    for (auto &op : setup.getRegion(0).front().without_terminator()) {
      auto v = op.getOperand(0);
      auto role = op.getAttrOfType<StringAttr>("access").getValue();
      StringRef mode = "record";
      if (role == "window-left")
        mode = p.stride(v, 3) == 1 && p.tile[0] <= p.tile[1]
                   ? "transpose-window-dma"
                   : "window-pack";
      else if (role == "window-right")
        mode = p.stride(v, 3) == 1 && intentdsl::kir::rowScale(p.dot)
                   ? "mapped-filter-major-dma"
                   : "mapped-row-pack";
      else if (role == "row-coefficient" || role == "row-prefix")
        mode = "shared-k-value";
      else if (role == "left-panel" || role == "right-panel") {
        bool lhs = role == "left-panel";
        auto k = p.axis(lhs ? "lhs_k" : "rhs_k"),
             batch = p.axis(lhs ? "lhs_batch" : "rhs_batch"),
             f = p.freeAxis(v, k, batch);
        auto inner = lhs ? k : f;
        if (lhs && p.routed() && p.stride(v, k) == 1)
          mode = "routed-row-dma";
        else if (!lhs && p.stride(v, k) == 1 && primitive != "f32-outer")
          mode = "filter-major-dma";
        else
          mode = p.stride(v, inner) == 1 ? "pitched-dma" : "indexed-pack";
      } else if (role == "quant-left")
        mode = p.stride(v, 1) == 1 ? "pitched-dma" : "indexed-pack";
      else if (role.starts_with("left-") || role.starts_with("right-"))
        mode = "group-scale";
      if (!field(op, "bangc.transfer", b.getStringAttr(mode)))
        return op.emitError("illegal panel transfer projection");
    }
    auto dst = p.domain.getOperand(0);
    bool contiguous = p.stride(dst, p.rank(dst) - 1) == 1 && !p.routed();
    if (p.rank(dst) == 1 &&
        llvm::any_of(p.domain.getRegion(0).front(), [](Operation &op) {
          return op.getName().stripDialect() == "add_columns";
        }))
      contiguous = false;
    // The transfer conversion fuses a single final narrowing only. A later
    // bias consumes the rounded value and must retain its scalar SSA type.
    for (auto &step : p.domain.getRegion(0).front())
      if (step.getName().stripDialect() == "convert_matrix") {
        auto type = mlir::cast<mlir::ShapedType>(step.getOperand(0).getType())
                        .getElementType();
        auto result = mlir::cast<mlir::ShapedType>(step.getResult(0).getType())
                          .getElementType();
        contiguous &=
            step.getNextNode() &&
            step.getNextNode()->getName().stripDialect() == "publish_matrix" &&
            type.isF32() && (result.isF16() || result.isBF16());
      }
    auto *convert = p.fold.getNextNode();
    auto *publish = convert ? convert->getNextNode() : nullptr;
    bool routedNarrow =
        p.routed() && p.stride(dst, p.rank(dst) - 1) == 1 && convert &&
        convert->getName().stripDialect() == "convert_matrix" &&
        convert->getOperand(0) == p.fold.getResult(0) && publish &&
        mlir::cast<mlir::ShapedType>(p.fold.getResult(0).getType()).getElementType().isF32() &&
        publish->getName().stripDialect() == "publish_matrix" &&
        mlir::cast<mlir::MemRefType>(dst.getType()).getElementType().isBF16();
    auto writeback = contiguous     ? "pitched-dma"
                     : routedNarrow ? "routed-row-dma"
                                    : "indexed-store";
    if (!field(setup, "bangc.writeback", b.getStringAttr(writeback)))
      return setup.emitError(
          "output projection disagrees with native transfer");
  }
  if (phase >= 2 &&
      !field(setup, "bangc.entry", b.getStringAttr("intent_enqueue")))
    return setup.emitError("missing matrix queue entry");
  return success();
}
} // namespace CI_UNIT_NAMESPACE
