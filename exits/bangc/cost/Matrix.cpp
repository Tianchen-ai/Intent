#include "intentdsl/levels/execution/block/ir/Contraction.h"
namespace CI_UNIT_NAMESPACE {
void panelCost(compilerinfra::Unit &u) {
  u.costModels.push_back(
      {{"matrix-transfer", "levels/execution/block::contraction", "1",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx) {
          auto t = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
          compilerinfra::CostEstimate estimate;
          estimate.scope = "one panel";
          estimate.bottleneck = "matrix supply";
          for (auto &math : op->getRegion(0).front()) {
            auto kind = math.getName().stripDialect();
            if (kind != "contract" && kind != "window_contract")
              continue;
            auto bytes =
                mlir::cast<mlir::MemRefType>(math.getOperand(0).getType())
                    .getElementTypeBitWidth() /
                8;
            double packing = t[1] * t[2] * bytes;
            if (kind == "window_contract")
              packing += t[1] * t[2] * sizeof(float) + 2 * t[2] * sizeof(float);
            estimate.terms["local_filter_pack_bytes"] = {packing, "B"};
            double transfers = kind == "window_contract" ? 4 : 2;
            estimate.terms["pitched_input_transfers"] = {transfers, "count"};
            auto latency = intentdsl::quantity(ctx, "dma_in_latency", "ns");
            auto bandwidth = intentdsl::quantity(ctx, "dma_in_bw", "B/ns");
            if (latency && bandwidth)
              estimate.timeNs =
                  transfers * *latency +
                  (packing + t[0] * t[2] * bytes) / *bandwidth;
          }
          return estimate;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
