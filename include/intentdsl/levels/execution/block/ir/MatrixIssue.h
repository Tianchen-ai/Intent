#pragma once
#include "intentdsl/levels/algorithm/kir/ir/NativeDot.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
namespace intentdsl::block {
// The exit provides the primitive's native fragment geometry. This estimate
// counts instruction groups for one output panel and never prunes a choice.
inline compilerinfra::CostEstimate
matrixIssues(mlir::Operation *domain,
             const compilerinfra::StructuralCandidate &candidate,
             compilerinfra::CostContext &ctx, int m, int n, int k,
             int warpRowCap = 4) {
  auto t = mlir::cast<mlir::DenseI64ArrayAttr>(candidate.value);
  bool isSplit = t.size() == 4 && (t[3] == 1 || t[3] == 2);
  compilerinfra::CostEstimate estimate;
  estimate.scope = "one panel";
  estimate.bottleneck = "matrix instructions";
  bool records = false;
  bool groupedRhs = false;
  bool quadFlat64 = false;
  double blockElems = 256.0;
  for (mlir::Operation &o : domain->getRegion(0).front()) {
    auto kind = o.getName().stripDialect();
    if ((kind == "contract" || kind == "window_contract" ||
         kind == "scaled_contract") &&
        kir::requiresSourceDot(
            &o, mlir::cast<mlir::ShapedType>(o.getOperand(0).getType())
                    .getElementType())) {
      // Widened products and both input panels are live before the K reduction.
      // Model traffic to local memory when that working set spills.
      double mRows = t[0], nCols = t[1], depth = t[2];
      double products = mRows * nCols * depth;
      double words = products + mRows * depth + nCols * depth + mRows * nCols;
      estimate.bottleneck = "FP32 product and reduction";
      estimate.terms["simt_products"] = {products, "count"};
      estimate.terms["simt_live_words"] = {words, "reg32"};
      auto f = ctx.target.services.get<CallFacts>();
      double sms = std::max<int64_t>(1, f ? f->participants : 1);
      if (auto rate = quantity(ctx, "compute", "lane/ns"))
        estimate.timeNs = 2. * products * sms / *rate;
      auto machine = ctx.target.services.get<OwnerMachine>();
      if (machine && machine->residentRegisters > 0) {
        double spill = 2. * sizeof(uint32_t) *
                       std::max(0., words - machine->residentRegisters);
        estimate.terms["simt_spill_bytes"] = {spill, "B"};
        if (auto bw = quantity(ctx, "read_1", "B/ns"))
          estimate.timeNs = estimate.timeNs.value_or(0.) + spill * sms / *bw;
      }
      return estimate;
    }
    if (kind == "q4k_dot") {
      records = true;
    } else if (kind == "quant_contract") {
      if (const auto *spec = intentdsl::kir::quantFormatSpec(o); spec) {
        if (spec->roleLayout == intentdsl::kir::QuantRoleLayout::RecordGemv) {
          records = true;
          blockElems = spec->blockElements;
          quadFlat64 = spec->blockElements == 256 && spec->numBitPlanes == 2 &&
                       spec->subScaleOffset >= 0 &&
                       spec->subBlockElements == 16 &&
                       spec->planes[0].halfStride == 64 &&
                       spec->planes[0].quadByteStep == 32 &&
                       spec->planes[0].quadShiftMode == 0 &&
                       spec->planes[1].halfStride == 32 &&
                       spec->planes[1].quadByteStep == 0 &&
                       spec->planes[1].quadShiftMode == 1;
        } else if (spec->roleLayout ==
                       intentdsl::kir::QuantRoleLayout::GroupedWeightRhs &&
                   !isSplit) {
          groupedRhs = true;
          blockElems = spec->blockElements;
        }
      }
    }
  }
  if (records) {
    auto mt = ctx.target.services.get<intentdsl::block::MatrixTarget>();
    bool flat64 = quadFlat64 && mt && mt->flatRecordBitPlanes;
    double decodeWidth = flat64 ? 64.0 : blockElems;
    double lanes = double(t[0]) * decodeWidth;
    estimate.terms["record_decode_lanes"] = {lanes, "count"};
    if (t[0] > warpRowCap) {
      double crossWarpBytes =
          double(t[0] - warpRowCap) * blockElems * 8.0;
      if (auto sbw = intentdsl::quantity(ctx, "read_1_shared_bw", "B/ns")) {
        auto f = ctx.target.services.get<intentdsl::CallFacts>();
        double smBw = *sbw / std::max<int64_t>(1, f ? f->participants : 1);
        estimate.timeNs = crossWarpBytes / smBw;
      } else if (auto rate = intentdsl::quantity(ctx, "compute", "lane/ns")) {
        double pressure = double(t[0]) / double(warpRowCap);
        estimate.timeNs = lanes * 16.0 * pressure / *rate;
      }
    } else if (auto rate = intentdsl::quantity(ctx, "compute", "lane/ns")) {
      estimate.timeNs = lanes * 16.0 / *rate;
    }
  } else {
    estimate.terms["native_matrix_fragments"] = {
        std::ceil(double(t[0]) / m) * std::ceil(double(t[1]) / n) *
            std::ceil(double(t[2]) / k),
        "count"};
    if (groupedRhs) {
      // Grouped packed-weight contractions decode (BN * BK) RHS lanes per
      // panel in registers; tiles that reuse each decoded weight panel across
      // more M-rows reduce total dequantization traffic and register pressure.
      double rhsElems = double(t[1]) * double(t[2]);
      double pressure = std::max(1.0, rhsElems / 8192.0);
      double groupAlign =
          (t[2] > 0 && blockElems >= double(t[2]))
              ? (blockElems / double(t[2]))
              : 1.5;
      estimate.terms["grouped_rhs_decode_lanes"] = {rhsElems, "count"};
      if (auto sbw = intentdsl::quantity(ctx, "read_1_shared_bw", "B/ns")) {
        auto f = ctx.target.services.get<intentdsl::CallFacts>();
        double smBw = *sbw / std::max<int64_t>(1, f ? f->participants : 1);
        estimate.timeNs =
            rhsElems * 2.0 * pressure * pressure * groupAlign / smBw;
      } else if (auto rate = intentdsl::quantity(ctx, "compute", "lane/ns")) {
        auto f = ctx.target.services.get<intentdsl::CallFacts>();
        double sms = std::max<double>(1.0, f ? f->participants : 1);
        estimate.timeNs = rhsElems * 6.0 * pressure * pressure * groupAlign *
                          sms / *rate;
      }
    }
  }
  return estimate;
}
} // namespace intentdsl::block
