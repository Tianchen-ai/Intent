#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
namespace CI_UNIT_NAMESPACE {
void registerBlockCost(compilerinfra::Unit &u) {
  u.costModels.push_back(
      {{"register-block", "levels/execution/cpu::matrix-partition", "2",
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
          auto tail = g->m % t[0];
          double fringe =
              tail ? (tail - 1) * columns * g->k /
                         std::max(1., std::ceil(double(g->m) / t[0]))
                   : 0.;
          conversions += fringe;
          e.terms["tail_rhs_reloads"] = {fringe, "instruction"};
          e.terms["broadcast_and_widen"] = {conversions, "instruction"};
          auto read = intentdsl::quantity(ctx, "read_1_private_bw", "B/ns"),
               write = intentdsl::quantity(ctx, "write_1_private_bw", "B/ns"),
               rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
          if (read && write && rate) {
            auto workers =
                ctx.target.services.get<intentdsl::cpu::RowMachine>()->workers;
            e.timeNs = (spill / 2) * workers * (1 / *read + 1 / *write) +
                       conversions * widths / *rate;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});

  u.costModels.push_back(
      {{"q4k-simd-row", "levels/execution/cpu::matrix-partition", "1",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx)
            -> llvm::Expected<compilerinfra::CostEstimate> {
          const auto t = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
          auto &dot = intentdsl::cpu::calculation(*op);
          auto g = intentdsl::kir::contractionShape(
              dot, op->getOperand(0),
              *ctx.target.services.get<intentdsl::CallFacts>());
          if (!g)
            return g.takeError();
          compilerinfra::CostEstimate e;
          e.scope = "one output task";
          e.bottleneck = "SIMD nibble unpack and dot";
          double subgroups = double(g->k) / 32.;
          double nibbleOps = t[0] * subgroups * 64.;
          e.terms["nibble_unpack_ops"] = {nibbleOps, "FLOP"};
          auto rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
          if (rate) {
            auto workers =
                ctx.target.services.get<intentdsl::cpu::RowMachine>()->workers;
            e.timeNs = (nibbleOps * workers) / *rate;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});

  u.costModels.push_back(
      {{"brgemm-tile", "levels/execution/cpu::matrix-partition", "1",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx)
            -> llvm::Expected<compilerinfra::CostEstimate> {
          const auto t = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
          auto *caps = ctx.target.services.get<intentdsl::cpu::MatrixMachine>();
          const auto &impl = caps->contracts[t[5]];
          auto &dot = intentdsl::cpu::calculation(*op);
          auto g = intentdsl::kir::contractionShape(
              dot, op->getOperand(0),
              *ctx.target.services.get<intentdsl::CallFacts>());
          if (!g)
            return g.takeError();
          compilerinfra::CostEstimate e;
          e.scope = "one output task";
          auto read = intentdsl::quantity(ctx, "read_1_private_bw", "B/ns"),
               write = intentdsl::quantity(ctx, "write_1_private_bw", "B/ns"),
               rate = intentdsl::quantity(ctx, "vector", "FLOP/ns"),
               launch = intentdsl::quantity(ctx, "launch", "ns");
          auto workers =
              ctx.target.services.get<intentdsl::cpu::RowMachine>()->workers;

          if (impl.dtype == "q4k") {
            e.bottleneck = "per-subgroup ukernel call and stack nibble unpack";
            double subgroups = double(g->k) / 32.;
            double unpackBytes = double(t[0]) * g->k * 2.;
            e.terms["subgroup_ukernel_calls"] = {subgroups, "count"};
            e.terms["nibble_stack_bytes"] = {unpackBytes, "B"};
            if (read && write) {
              double callCost = launch ? std::min(*launch * 0.05, 25.) : 20.;
              e.timeNs = subgroups * callCost +
                         (unpackBytes / 2.) * workers *
                             (1. / *read + 1. / *write);
            }
            return e;
          }

          if (impl.dtype == "s8") {
            bool amx = ctx.target.features.count("amx_int8") > 0;
            e.bottleneck = amx ? "AMX int8 tile unit" : "VNNI JIT microkernel";
            double tileBytes = double(t[0]) * t[1] * 4.;
            e.terms["tile_buffer_bytes"] = {tileBytes, "B"};
            if (read && write) {
              e.timeNs =
                  tileBytes * workers * (1. / *read + 1. / *write) *
                  (amx ? 0.25 : 1.0);
            }
            return e;
          }

          e.bottleneck = "JIT AVX-512 BRGeMM microkernel";
          double columns = std::ceil(double(t[1]) / t[3]);
          double live = t[0] * columns + columns + 2;
          double spill = std::max(0., live - (t[3] == 16 ? 32 : 16)) * t[3] *
                         4 * 2 * g->k;
          double inTileConvert = 0.;
          if (impl.dtype == "f16" &&
              !ctx.target.features.count("amx_fp16")) {
            // Without AMX-FP16, AVX-512 converts f16->f32 inside the O(M*N*K)
            // tile loop rather than accumulating native f16->f32 products.
            inTileConvert = double(g->k) * (t[0] + columns);
          }
          double tileBytes = double(t[0]) * t[1] * 4.;
          e.terms["live_vectors"] = {live, "count"};
          e.terms["spill_bytes_per_task"] = {spill, "B"};
          e.terms["in_tile_f16_widen"] = {inTileConvert, "instruction"};
          e.terms["tile_buffer_bytes"] = {tileBytes, "B"};
          if (read && write && rate) {
            double widths = workers * t[3];
            e.timeNs = (spill / 2. + tileBytes / 2.) * workers *
                           (1. / *read + 1. / *write) +
                       (inTileConvert * widths) / *rate;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
