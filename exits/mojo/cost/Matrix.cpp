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
                 live = t[0] * columns + columns +
                        (t[4] == 2 ? double(t[0]) + 2. : 1.);
          double spill = std::max(0., live - (t[3] == 16 ? 32 : 16)) * t[3] *
                         4 * 2 * g->k / (t[4] == 2 ? 4 : 1);
          e.terms["live_vectors"] = {live, "count"};
          e.terms["spill_bytes_per_task"] = {spill, "B"};
          double widths =
              ctx.target.services.get<intentdsl::cpu::RowMachine>()->workers *
              t[3];
          double l1Loads = (t[4] == 2 ? (g->k / 4.) * (t[0] + columns)
                                      : g->k * (t[0] + columns));
          bool needsWiden = false;
          if (t[4] != 2 && t[4] != 3 &&
              dot.getName().stripDialect() != "scaled_contract") {
            auto inBytes =
                mlir::cast<mlir::MemRefType>(dot.getOperand(0).getType())
                    .getElementTypeBitWidth() /
                8;
            needsWiden = inBytes < 4;
          }
          double conversions =
              needsWiden ? double(g->k) * (columns + double(t[0]) / t[3]) : 0.;
          auto tail = g->m % t[0];
          double fringe =
              tail ? (tail - 1) * columns * g->k /
                         std::max(1., std::ceil(double(g->m) / t[0]))
                   : 0.;
          auto colTail = g->n % t[1];
          if (colTail) {
            double effMr = double(std::min<int64_t>(g->m, t[0]));
            double colVecs = std::ceil(double(colTail) / t[3]);
            double kSteps = t[4] == 2 ? double(g->k) / 4. : double(g->k);
            fringe +=
                ((effMr - 1.) * colVecs + effMr * std::max(0., colVecs - 1.)) *
                kSteps / std::max(1., std::ceil(double(g->n) / t[1]));
          }
          conversions += fringe;
          e.terms["l1_panel_loads"] = {l1Loads, "instruction"};
          e.terms["tail_rhs_reloads"] = {fringe, "instruction"};
          e.terms["broadcast_and_widen"] = {conversions, "instruction"};
          auto read = intentdsl::quantity(ctx, "read_1_private_bw", "B/ns"),
               write = intentdsl::quantity(ctx, "write_1_private_bw", "B/ns"),
               rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
          if (read && write && rate) {
            auto workers =
                ctx.target.services.get<intentdsl::cpu::RowMachine>()->workers;
            e.timeNs = (spill / 2) * workers * (1 / *read + 1 / *write) +
                       ((l1Loads * 0.5 + conversions) * widths) / *rate;
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
          auto *caps = ctx.target.services.get<intentdsl::cpu::MatrixMachine>();
          bool isVnni =
              caps && t.size() > 5 &&
              caps->contracts[t[5]].id == "exits/mojo::q4k-vnni";
          compilerinfra::CostEstimate e;
          e.scope = "one output task";
          e.bottleneck = "SIMD nibble unpack and dot";
          double subgroups = double(g->k) / 32.;
          double nibbleOps = t[0] * subgroups * (isVnni ? 16. : 64.);
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
               launch = intentdsl::quantity(ctx, "launch", "ns"),
               sync = intentdsl::quantity(ctx, "sync", "ns");
          auto workers =
              ctx.target.services.get<intentdsl::cpu::RowMachine>()->workers;
          double jobs = std::ceil(double(g->m) / t[0]) *
                        std::ceil(double(g->n) / t[1]) * g->batches;
          double waves = std::max(1., std::ceil(jobs / workers));
          double planNs = sync ? *sync * 1.5 : 1000.;
          double ffiCallNs = sync ? *sync * 0.15 : 100.;
          double amxEnterLeaveNs =
              (launch ? *launch * 4.0 : 11000.) +
              (sync ? *sync * 2.0 : 1400.);

          if (impl.dtype == "q4k") {
            bool amx = ctx.target.features.count("amx_int8") > 0;
            e.bottleneck = "per-subgroup ukernel call and stack nibble unpack";
            double subgroups = double(g->k) / 32.;
            double unpackBytes = double(t[0]) * g->k * 2.;
            double amxCtx = amx ? amxEnterLeaveNs : planNs;
            e.terms["subgroup_ukernel_calls"] = {subgroups, "count"};
            e.terms["nibble_stack_bytes"] = {unpackBytes, "B"};
            if (read && write) {
              double callCost = launch ? std::min(*launch * 0.05, 25.) : 20.;
              e.timeNs = amxCtx / waves + subgroups * callCost +
                         (unpackBytes / 2.) * workers *
                             (1. / *read + 1. / *write);
            }
            return e;
          }

          if (impl.dtype == "s8" || impl.dtype == "bf16") {
            bool amx = impl.dtype == "s8"
                           ? (ctx.target.features.count("amx_int8") > 0)
                           : (ctx.target.features.count("amx_bf16") > 0);
            e.bottleneck = amx ? "AMX tile unit and context transition"
                               : "VNNI/BF16 JIT microkernel";
            double columns = std::ceil(double(t[1]) / t[3]);
            double tileBytes = double(t[0]) * t[1] * 4.;
            double amxCtx = amx ? amxEnterLeaveNs : planNs;
            e.terms["tile_buffer_bytes"] = {tileBytes, "B"};
            e.terms["ukernel_setup_ns"] = {amxCtx / waves + ffiCallNs, "ns"};
            if (read && write && rate) {
              double widths = workers * t[3];
              e.timeNs = amxCtx / waves + ffiCallNs +
                         tileBytes * workers * (1. / *read + 1. / *write) +
                         (double(t[0]) * columns * widths) / *rate;
            }
            return e;
          }

          e.bottleneck = "JIT AVX-512 BRGeMM microkernel";
          double columns = std::ceil(double(t[1]) / t[3]);
          double live = t[0] * columns + columns + 2;
          double spill = std::max(0., live - (t[3] == 16 ? 32 : 16)) * t[3] *
                         4 * 2 * g->k;
          double l1Loads = double(g->k) * (t[0] + columns);
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
          e.terms["l1_panel_loads"] = {l1Loads, "instruction"};
          e.terms["in_tile_f16_widen"] = {inTileConvert, "instruction"};
          e.terms["tile_buffer_bytes"] = {tileBytes, "B"};
          e.terms["ukernel_setup_ns"] = {planNs / waves + ffiCallNs, "ns"};
          if (read && write && rate) {
            double widths = workers * t[3];
            e.timeNs = planNs / waves + ffiCallNs +
                       (spill / 2. + tileBytes / 2.) * workers *
                           (1. / *read + 1. / *write) +
                       ((l1Loads * 0.5 + inTileConvert) * widths) / *rate;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});

  u.costModels.push_back(
      {{"simd-row-primitive", "levels/execution/cpu::partition", "2",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx)
            -> llvm::Expected<compilerinfra::CostEstimate> {
          auto [m, n] =
              ctx.target.services.get<intentdsl::CallFacts>()->geometry(op);
          auto d = mlir::cast<mlir::DictionaryAttr>(c.value);
          if (d.getAs<mlir::StringAttr>("walk") == "destination")
            std::swap(m, n);
          if (d.contains("member_tasks")) {
            n = m * n;
            m = 1.;
          }
          double lanes =
              ctx.target.services.get<intentdsl::cpu::RowMachine>()->vectorBits /
              32.;
          int64_t unroll = d.getAs<mlir::IntegerAttr>("unroll")
                               ? d.getAs<mlir::IntegerAttr>("unroll").getInt()
                               : 1;
          double sweeps = m * std::ceil(n / lanes);
          double treeDepth = std::log2(std::max(1., lanes));
          unsigned uniforms = 0, moments = 0, scans = 0, reds = 0, exps = 0,
                   divs = 0, reads = 0, writes = 0, bodyValues = 0;
          op->walk([&](mlir::Operation *i) {
            auto k = i->getName().stripDialect();
            uniforms += (k == "uniform");
            moments += (k == "welford");
            scans += (k == "scan_sum");
            reds += (k == "sum" || k == "max_num");
            exps += (k == "exp");
            divs += (k == "div" || k == "sqrt" || k == "rsqrt");
            reads += (k == "read" || k == "read_owner");
            writes += (k == "write");
            if (i != op && k != "end" && k != "constant" &&
                i->getNumResults() == 1)
              ++bodyValues;
          });
          double archVecRegs = lanes >= 16. ? 32. : 16.;
          double polyRegs = exps > 0 ? 6. : 0.;
          double opScratch = exps > 0 ? 2. : (divs > 0 ? 1. : 0.);
          double lanePeak = std::min<double>(
              10., std::max<double>(1., double(reads) + opScratch +
                                            0.25 * double(bodyValues)));
          double overlap =
              unroll > 1 ? (reds > 0 ? 2. : 4.) : 1.;
          double liveVecs =
              polyRegs + double(unroll) * double(reds) + overlap * lanePeak;
          double spillVecs = std::max(0., liveVecs - archVecRegs);
          double spillBytes = spillVecs * lanes * 4. * 2. * sweeps;
          double unrolledBlocks =
              m * std::floor(n / (lanes * double(unroll)));
          double remBlocks =
              sweeps - unrolledBlocks * double(unroll);
          double chainSteps =
              (unrolledBlocks + remBlocks) * (4. * double(reds) + 1.5) +
              (unroll > 1 && unrolledBlocks == 0. && sweeps > 0. ? m * 4. : 0.);
          compilerinfra::CostEstimate e;
          e.scope = "row task";
          e.bottleneck = spillBytes > 0. ? "SIMD register spill"
                                         : "SIMD microkernel and loop issue";
          double vecSteps =
              sweeps * (10. * uniforms + 4. * moments + treeDepth * scans) +
              m * treeDepth * moments + chainSteps;
          e.terms["unroll_factor"] = {double(unroll), "count"};
          e.terms["live_vectors"] = {liveVecs, "count"};
          e.terms["spill_bytes"] = {spillBytes, "B"};
          e.terms["reduction_and_loop_steps"] = {chainSteps, "instruction"};
          e.terms["vector_steps"] = {vecSteps, "instruction"};
          auto rate = intentdsl::quantity(ctx, "vector", "FLOP/ns"),
               rdPriv = intentdsl::quantity(ctx, "read_1_private_bw", "B/ns"),
               wrPriv = intentdsl::quantity(ctx, "write_1_private_bw", "B/ns");
          double effRate = rate.value_or(std::max(1., lanes * 2.));
          double t = (vecSteps * lanes) / effRate;
          if (spillBytes > 0. && rdPriv && wrPriv)
            t += (spillBytes / 2.) * (1. / *rdPriv + 1. / *wrPriv);
          else if (spillBytes > 0.)
            t += spillBytes / 16.;
          e.timeNs = t;
          return e;
        }},
       COMPILERINFRA_SOURCE});

  u.costModels.push_back(
      {{"simd-tile-transpose", "levels/execution/cpu::permutation", "1",
        [](mlir::Operation *op, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx)
            -> llvm::Expected<compilerinfra::CostEstimate> {
          auto [m, n] =
              ctx.target.services.get<intentdsl::CallFacts>()->geometry(op);
          auto d = mlir::cast<mlir::DictionaryAttr>(c.value);
          compilerinfra::CostEstimate e;
          e.scope = "tile transpose";
          e.bottleneck = "SIMD shuffle network";
          double blocks8x8 = d.getAs<mlir::StringAttr>("walk") == "tile"
                                 ? std::ceil(m / 8.) * std::ceil(n / 8.)
                                 : 0.;
          double shuffles = blocks8x8 * 24.;
          e.terms["subblock_8x8_count"] = {blocks8x8, "count"};
          e.terms["register_shuffles"] = {shuffles, "instruction"};
          if (auto rate = intentdsl::quantity(ctx, "vector", "FLOP/ns"))
            e.timeNs = (shuffles * 8.) / *rate;
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
