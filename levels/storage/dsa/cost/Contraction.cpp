#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
void matrixSupplyCost(Unit &u) {
  u.costModels.push_back(
      {{"panel-transfers", u.qualify("panel-supply"), "1",
        [](mlir::Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          intentdsl::block::MatrixAccess a(mlir::cast<mlir::func::FuncOp>(op));
          auto t = a.tile, d = a.shape;
          double jobs = std::ceil(double(d[0]) / t[0]) *
                        std::ceil(double(d[1]) / t[1]) * d[3];
          double workers = mlir::cast<mlir::IntegerAttr>(c.value).getInt(),
                 waves = std::ceil(jobs / workers);
          double panels = std::ceil(double(d[2]) / t[2]),
                 bytes = 2 * t[2] * (t[0] + t[1]), transactions = 2 * panels;
          CostEstimate e;
          e.scope = "full_call";
          e.bottleneck = "panel supply";
          e.terms["waves"] = {waves, "count"};
          e.terms["bytes_per_panel"] = {bytes, "B"};
          e.terms["transactions"] = {transactions, "count"};
          auto bw = intentdsl::quantity(ctx, "read_1", "B/ns"),
               sync = intentdsl::quantity(ctx, "sync", "ns");
          if (bw)
            e.timeNs = waves * (bytes * panels / *bw +
                                (sync ? transactions * *sync : 0));
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
