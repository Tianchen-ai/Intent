#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/MatrixAccess.h"
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
          bool isCluster = (t[0] == 512 && t[1] == 512);
          double clusterWaveFactor = 1.0;
          if (isCluster) {
            if (auto *f = ctx.target.services.get<intentdsl::CallFacts>()) {
              if (auto sh = intentdsl::block::matrixShape(op, *f)) {
                double jobs = sh->batches *
                              std::ceil(double(sh->m) / double(t[0])) *
                              std::ceil(double(sh->n) / double(t[1]));
                double cWaves = std::ceil(jobs / 4.0);
                double pWaves = std::max(
                    1.0, std::ceil(jobs / std::max<int64_t>(1, f->participants)));
                clusterWaveFactor = cWaves / pWaves;
              }
            }
          }
          for (auto &math : op->getRegion(0).front()) {
            auto kind = math.getName().stripDialect();
            if (kind == "quant_contract") {
              const auto *qSpec = intentdsl::kir::quantFormatSpec(math);
              if (!qSpec ||
                  qSpec->roleLayout == intentdsl::kir::QuantRoleLayout::RecordGemv)
                continue;
              auto latency = intentdsl::quantity(ctx, "dma_in_latency", "ns");
              auto bandwidth = intentdsl::quantity(ctx, "dma_in_bw", "B/ns");
              if (qSpec->roleLayout ==
                  intentdsl::kir::QuantRoleLayout::BlockScaledPair) {
                double decElems = isCluster
                                      ? ((t[0] / 4.0 + t[1] / 4.0) * t[2])
                                      : ((t[0] + t[1]) * t[2]);
                double coreFlops = isCluster
                                       ? (2.0 * (t[0] / 2.0) * (t[1] / 2.0) * t[2])
                                       : (2.0 * t[0] * t[1] * t[2]);
                estimate.terms["local_filter_pack_bytes"] = {decElems * 2.0, "B"};
                estimate.terms["pitched_input_transfers"] = {4.0, "count"};
                if (latency && bandwidth) {
                  double decNs = decElems * 0.45 + coreFlops / 2560.0 +
                                 4.0 * *latency;
                  estimate.timeNs = isCluster ? (clusterWaveFactor * decNs)
                                              : decNs;
                }
              } else if (qSpec->roleLayout ==
                         intentdsl::kir::QuantRoleLayout::GroupedWeightRhs) {
                double rhsBytes = t[1] * t[2] * 0.5625;
                double lhsBytes = t[0] * t[2] * 2.0 / 4.0;
                double coreFlops = 2.0 * t[0] * t[1] * t[2];
                estimate.terms["local_filter_pack_bytes"] = {rhsBytes, "B"};
                estimate.terms["pitched_input_transfers"] = {3.0, "count"};
                if (latency && bandwidth)
                  estimate.timeNs = 3.0 * *latency +
                                    (rhsBytes + lhsBytes) / *bandwidth +
                                    (t[1] * t[2]) * 0.18 + coreFlops / 2560.0;
              }
              continue;
            }
            if (kind != "contract" && kind != "window_contract")
              continue;
            auto bytes =
                llvm::divideCeil(
                    mlir::cast<mlir::MemRefType>(math.getOperand(0).getType())
                        .getElementTypeBitWidth(),
                    8u);
            if (isCluster) {
              double sramStage = (t[0] + t[1]) * t[2] * bytes;
              double coreFlops = 2.0 * (t[0] / 2.0) * (t[1] / 2.0) * t[2];
              estimate.terms["local_filter_pack_bytes"] = {
                  (t[1] / 2.0) * t[2] * bytes, "B"};
              estimate.terms["pitched_input_transfers"] = {2.0, "count"};
              auto dramBw = intentdsl::quantity(ctx, "read_1", "B/ns");
              auto latency = intentdsl::quantity(ctx, "dma_in_latency", "ns");
              if (dramBw && latency) {
                double mpuNs = sramStage / (*dramBw / 4.0);
                double ipuNs = coreFlops / 2400.0;
                double panelNs = 2.0 * *latency + std::max(mpuNs, ipuNs);
                estimate.timeNs = clusterWaveFactor * panelNs;
              }
              continue;
            }
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
  u.costModels.push_back(
      {{"cnnl-matmul-cost", "levels/storage/dsa::panel-supply", "1",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &,
           compilerinfra::CostContext &ctx) {
          intentdsl::block::MatrixAccess a(mlir::cast<mlir::func::FuncOp>(op));
          auto d = a.shape;
          double M = d[0], N = d[1], K = d[2], B = d[3];
          double elemBytes =
              llvm::divideCeil(
                  mlir::cast<mlir::MemRefType>(a.dot.getOperand(0).getType())
                      .getElementTypeBitWidth(),
                  8u);
          double totalBytes = B * (M * K + K * N + M * N) * elemBytes;
          double flops = 2.0 * B * M * N * K;
          compilerinfra::CostEstimate e;
          e.scope = "full_call";
          e.bottleneck = "external library";
          e.terms["bytes"] = {totalBytes, "B"};
          e.terms["flops"] = {flops, "count"};
          auto bw = intentdsl::quantity(ctx, "read_1", "B/ns"),
               launch = intentdsl::quantity(ctx, "launch", "ns");
          if (bw && launch) {
            double peakOpsPerNs = elemBytes <= 2.0 ? 48000.0 : 12000.0;
            e.timeNs = 1.8 * B * (*launch) + totalBytes / (*bw) +
                       flops / peakOpsPerNs;
            e.rows["0"] = {e.terms, e.timeNs};
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
