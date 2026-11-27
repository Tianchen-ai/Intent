#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
void matrixSupplyCost(Unit &u) {
  u.costModels.push_back(
      {{"panel-transfers", u.qualify("panel-supply"), "1",
        [](mlir::Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          if (mlir::isa<mlir::StringAttr>(c.value)) {
            auto ms = ctx.target.services.get<intentdsl::dsa::MatrixStorage>();
            if (!ms || ms->libraryEstimator.empty())
              return llvm::createStringError(
                  "missing target library matrix estimator");
            return ctx.local(ms->libraryEstimator);
          }
          intentdsl::block::MatrixAccess a(
              op->getParentOfType<mlir::func::FuncOp>());
          auto t = a.tile, d = a.shape;
          double jobs = std::ceil(double(d[0]) / t[0]) *
                        std::ceil(double(d[1]) / t[1]) * d[3];
          auto ms = ctx.target.services.get<intentdsl::dsa::MatrixStorage>();
          auto plan = ms ? intentdsl::dsa::PanelPlan::read(c.value, a, *ms)
                         : std::nullopt;
          if (!plan)
            return llvm::createStringError("invalid matrix placement for estimation");
          bool isCluster = plan->placement == "shared-tile";
          double rawWorkers = plan->participants,
                 workers = isCluster ? std::max(1.0, rawWorkers / ms->coresPerCluster)
                                     : rawWorkers,
                 waves = std::ceil(jobs / workers),
                 active = std::min(workers, std::max(1.0, jobs));
          double inElemBytes =
              llvm::divideCeil(
                  mlir::cast<mlir::MemRefType>(a.dot.getOperand(0).getType())
                      .getElementTypeBitWidth(),
                  8u);
          double panels = std::ceil(double(d[2]) / t[2]),
                 bytes = inElemBytes * t[2] * (t[0] + t[1]),
                 transactions = 2 * panels;
          CostEstimate e;
          if (plan->placement == "shared-lhs") {
            // Only complete M tiles stage one LHS for the cluster. Tail rows
            // use the independent GDRAM path on each consuming core.
            double fullFraction = (d[0] / t[0]) / std::ceil(double(d[0]) / t[0]);
            double shared = inElemBytes * t[2] * t[0] * fullFraction;
            bytes -= shared * (1.0 - 1.0 / ms->coresPerCluster);
            e.terms["shared_lhs_bytes_per_panel"] = {shared, "B"};
          }
          e.scope = "full_call";
          e.bottleneck = "panel supply";
          e.terms["waves"] = {waves, "count"};
          e.terms["bytes_per_panel"] = {bytes, "B"};
          e.terms["transactions"] = {transactions, "count"};
          auto bw = intentdsl::quantity(ctx, "read_1", "B/ns"),
               dmaIn = intentdsl::quantity(ctx, "dma_in_bw", "B/ns"),
               sync = intentdsl::quantity(ctx, "sync", "ns"),
               launch = intentdsl::quantity(ctx, "launch", "ns");
          if (bw) {
            if (isCluster) {
              double perClusterBw = *bw / active;
              double dmaNs = bytes * panels / perClusterBw;
              double coreComputeNs =
                  (2.0 * (t[0] / 2.0) * (t[1] / 2.0) * t[2] * panels) / 2048.0;
              double waveNs = std::max(dmaNs, coreComputeNs) +
                              0.2 * std::min(dmaNs, coreComputeNs) +
                              (sync ? transactions * *sync : 0);
              e.timeNs = launch.value_or(0.0) + waves * waveNs;
            } else {
              double perWorkerBw = std::min(dmaIn.value_or(*bw), *bw / active);
              double computeNs = (2.0 * t[0] * t[1] * t[2] * panels) / 512.0;
              e.timeNs =
                  launch.value_or(0.0) +
                  waves * (bytes * panels / perWorkerBw + computeNs +
                           (sync ? transactions * *sync : 0));
            }
            if (e.timeNs)
              e.rows["0"] = {e.terms, e.timeNs};
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
