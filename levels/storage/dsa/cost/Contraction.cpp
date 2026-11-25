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
          intentdsl::block::MatrixAccess a(mlir::cast<mlir::func::FuncOp>(op));
          auto t = a.tile, d = a.shape;
          double jobs = std::ceil(double(d[0]) / t[0]) *
                        std::ceil(double(d[1]) / t[1]) * d[3];
          double workers = mlir::cast<mlir::IntegerAttr>(c.value).getInt(),
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
            double perWorkerBw = std::min(dmaIn.value_or(*bw), *bw / active);
            double computeNs = (2.0 * t[0] * t[1] * t[2] * panels) / 512.0;
            e.timeNs =
                launch.value_or(0.0) +
                waves * (bytes * panels / perWorkerBw + computeNs +
                         (sync ? transactions * *sync : 0));
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
