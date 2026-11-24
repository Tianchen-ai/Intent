#include "intentdsl/levels/execution/block/ir/Contraction.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
void matrixCost(Unit &u) {
  u.costModels.push_back(
      {{"tile-supply", u.qualify("contraction"), "1",
        [](mlir::Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto f = ctx.target.services.get<intentdsl::CallFacts>();
          auto shape = intentdsl::block::matrixShape(op, *f);
          if (!shape)
            return shape.takeError();
          auto tile = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
          double tm = tile[0], tn = tile[1], tk = tile[2];
          bool records =
              llvm::any_of(op->getRegion(0).front(), [](mlir::Operation &o) {
                return o.getName().stripDialect() == "q4k_dot";
              });
          double jobs = shape->batches * std::ceil(shape->m / tm) *
                        std::ceil(shape->n / tn);
          double panels =
              records ? std::ceil(shape->k / 256.0) : std::ceil(shape->k / tk);
          double panelBytes =
              records ? (tm * 144.0 + 256.0 * 4.0) : ((tm + tn) * tk * 2.0);
          double read =
              records ? (shape->batches * panels * (shape->m * 144.0 + 1024.0))
                      : (jobs * panels * panelBytes);
          double waves =
              std::ceil(jobs / std::max<int64_t>(1, f->participants));
          CostEstimate e;
          e.scope = "full_call";
          e.bottleneck = "matrix tile supply";
          e.terms["operand_bytes"] = {read, "B"};
          e.terms["output_tiles"] = {jobs, "count"};
          e.terms["matrix_panel_waves"] = {waves * panels, "count"};
          auto bw = intentdsl::quantity(ctx, "read_1", "B/ns"),
               startup = intentdsl::quantity(ctx, "launch", "ns");
          // A panel has two issue operations even when widening K leaves the
          // number of operand bytes unchanged; row-collective record dots stay
          // warp-local without shared-memory panel barriers.
          double issues = records ? 0.0 : (2 * waves * panels);
          e.terms["panel_load_issues"] = {issues, "count"};
          auto sync = intentdsl::quantity(ctx, "sync", "ns");
          double waveBytes =
              records ? (tm * 144.0 * std::max<int64_t>(1, f->participants) +
                         1024.0)
                      : (panelBytes * std::max<int64_t>(1, f->participants));
          if (bw && startup)
            e.timeNs = *startup + read / *bw +
                       waves * panels * waveBytes / *bw +
                       (sync ? issues * *sync : 0);
          const auto &est =
              ctx.target.services.get<intentdsl::block::MatrixTarget>()
                  ->estimate;
          if (!est.empty()) {
            auto local = ctx.local(est);
            if (!local)
              return local.takeError();
            e.terms.insert(local->terms.begin(), local->terms.end());
            if (local->timeNs && e.timeNs)
              *e.timeNs += waves * panels * *local->timeNs;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
