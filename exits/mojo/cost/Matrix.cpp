#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
namespace CI_UNIT_NAMESPACE {
void registerBlockCost(compilerinfra::Unit &u) {
  u.costModels.push_back(
      {{"register-block", "levels/execution/cpu::matrix-partition", "1",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx)
            -> llvm::Expected<compilerinfra::CostEstimate> {
          const auto t = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
          auto &dot = intentdsl::cpu::calculation(*op);
          compilerinfra::CostEstimate e;
          e.scope = "one output task";
          e.bottleneck = "SIMD register file";
          if (dot.getName().stripDialect() == "q4k_dot")
            return e;
          auto g = intentdsl::kir::contractionShape(
              dot, op->getOperand(0),
              *ctx.target.services.get<intentdsl::CallFacts>());
          if (!g)
            return g.takeError();
          double columns = std::ceil(double(t[1]) / t[3]),
                 live = t[0] * columns + columns + 1;
          double spill = std::max(0., live - (t[3] == 16 ? 32 : 16)) * t[3] *
                         4 * 2 * g->k / (t[4] == 2 ? 4 : 1);
          e.terms["live_vectors"] = {live, "count"};
          e.terms["spill_bytes_per_task"] = {spill, "B"};
          double widths =
              ctx.target.services.get<intentdsl::cpu::RowMachine>()->workers *
              t[3];
          double conversions = (t[4] == 2 ? 0. : g->k * (t[0] + columns));
          // Partial output blocks take the row-by-row vector loop. Each of
          // those rows reloads the RHS instead of sharing the full block's
          // load.
          auto tail = g->m % t[0];
          double fringe = tail ? (tail - 1) * columns * g->k /
                                     std::ceil(double(g->m) / t[0])
                               : 0.;
          conversions += fringe;
          e.terms["tail_rhs_reloads"] = {fringe, "instruction"};
          e.terms["broadcast_and_widen"] = {conversions, "instruction"};
          auto bw = intentdsl::quantity(ctx, "read_1_private_bw", "B/ns"),
               rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
          if (bw && rate)
            e.timeNs = spill / *bw + conversions * widths / *rate;
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
