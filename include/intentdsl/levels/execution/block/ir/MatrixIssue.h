#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
namespace intentdsl::block {
// The exit provides the primitive's native fragment geometry. This estimate
// counts instruction groups for one output panel and never prunes a choice.
inline compilerinfra::CostEstimate
matrixIssues(mlir::Operation *domain,
             const compilerinfra::StructuralCandidate &candidate,
             compilerinfra::CostContext &ctx, int m, int n, int k,
             int warpRowCap = 4) {
  auto t = mlir::cast<mlir::DenseI64ArrayAttr>(candidate.value);
  compilerinfra::CostEstimate estimate;
  estimate.scope = "one panel";
  estimate.bottleneck = "matrix instructions";
  bool records = false;
  double blockElems = 256.0;
  for (mlir::Operation &o : domain->getRegion(0).front()) {
    auto kind = o.getName().stripDialect();
    if (kind == "q4k_dot") {
      records = true;
    } else if (kind == "quant_contract") {
      if (const auto *spec = intentdsl::kir::quantFormatSpec(o);
          spec &&
          spec->roleLayout == intentdsl::kir::QuantRoleLayout::RecordGemv) {
        records = true;
        blockElems = spec->blockElements;
      }
    }
  }
  if (records) {
    double lanes = double(t[0]) * blockElems;
    estimate.terms["record_decode_lanes"] = {lanes, "count"};
    if (t[0] > warpRowCap) {
      double crossWarpBytes = double(t[0] - warpRowCap) * blockElems * 8.0;
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
  }
  return estimate;
}
} // namespace intentdsl::block
