#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
void matrixCost(Unit &u) {
  u.costModels.push_back(
      {{"matrix-work", u.qualify("matrix-partition"), "2",
        [](mlir::Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto f = ctx.target.services.get<intentdsl::CallFacts>();
          auto &dot = intentdsl::cpu::calculation(*op);
          auto g = intentdsl::kir::contractionShape(dot, op->getOperand(0), *f);
          if (!g)
            return g.takeError();
          auto t = mlir::cast<mlir::DenseI64ArrayAttr>(c.value);
          auto *caps = ctx.target.services.get<intentdsl::cpu::MatrixMachine>();
          double jobs = std::ceil(double(g->m) / t[0]) *
                        std::ceil(double(g->n) / t[1]) * g->batches;
          auto workers =
              ctx.target.services.get<intentdsl::cpu::RowMachine>()->workers;
          double waves = jobs > 0 ? std::ceil(jobs / workers) : 0.;
          int64_t packCode = t[4];
          double elemReadBytes = packCode == 3 ? 4. : 2.;
          if (dot.getName().stripDialect() == "contract") {
            auto inBytes =
                mlir::cast<mlir::MemRefType>(dot.getOperand(0).getType())
                    .getElementTypeBitWidth() /
                8.;
            elemReadBytes = packCode == 3 ? 4. : inBytes;
          } else if (dot.getName().stripDialect() == "scaled_contract") {
            elemReadBytes = 4.;
          }
          double reads = waves * (t[0] + t[1]) * g->k * elemReadBytes;
          double ops = waves * t[0] * std::ceil(double(t[1]) / t[3]) * g->k;
          if (packCode == 2 ||
              (t.size() == 6 && caps->contracts[t[5]].dtype == "s8"))
            ops /= 4.;
          double packBytes = 0, translationBytes = 0, scatteredBytes = 0;
          double widenOps = 0;
          if (dot.getName().stripDialect() == "window_contract") {
            auto slot = mlir::cast<mlir::BlockArgument>(dot.getOperand(0))
                            .getArgNumber();
            auto stride =
                (*f->arguments[slot].getAsObject()->getArray("strides"))[1]
                    .getAsInteger()
                    .value();
            double line =
                std::min(64., std::max(2., std::abs(double(stride)) * 2.));
            if (packCode) {
              packBytes =
                  double(g->batches) * g->k * ((line + 4) * g->m + 6 * g->n);
            } else {
              double colBlocks = std::ceil(double(g->n) / t[1]);
              packBytes = double(g->batches) * g->k * 4. * g->n;
              scatteredBytes =
                  double(g->batches) * g->m * g->k * line * colBlocks;
              widenOps =
                  waves * g->k * (std::ceil(double(t[1]) / t[3]) + t[0]) * t[3];
            }
          }
          if (dot.getName().stripDialect() == "contract") {
            auto b = dot.getOperand(1);
            auto slot = mlir::cast<mlir::BlockArgument>(b).getArgNumber();
            int k = dot.getAttrOfType<mlir::IntegerAttr>("rhs_k").getInt();
            int batch =
                dot.getAttrOfType<mlir::IntegerAttr>("rhs_batch").getInt();
            int column = 0;
            while (column == k || column == batch)
              ++column;
            auto strides = f->arguments[slot].getAsObject()->getArray("strides");
            auto step = (*strides)[k].getAsInteger().value();
            auto bytes = mlir::cast<mlir::MemRefType>(b.getType())
                             .getElementTypeBitWidth() /
                         8;
            double rhsBanks = batch < 0 ? 1. : double(g->batches);
            if (packCode == 3) {
              int lhsBatch =
                  dot.getAttrOfType<mlir::IntegerAttr>("lhs_batch").getInt();
              double lhsBanks = lhsBatch < 0 ? 1. : double(g->batches);
              double lhsPack =
                  g->routed ? double(g->m) * g->k * std::max(0., 4. - bytes)
                            : lhsBanks * g->m * g->k * (bytes + 4.);
              double colStride =
                  std::abs(double((*strides)[column].getAsInteger().value()));
              double extraOnce =
                  std::max(0., std::min(64., colStride * bytes) - bytes);
              packBytes =
                  lhsPack + rhsBanks * g->n * g->k * (bytes + 4. + extraOnce);
            } else if (packCode) {
              packBytes = 2. * g->n * g->k * rhsBanks * bytes;
            } else if (step * bytes >= 4096) {
              translationBytes = jobs * g->k * 64.;
            }
            if (!packCode) {
              double columnStride =
                  std::abs(double((*strides)[column].getAsInteger().value()));
              double extra =
                  std::max(0., std::min(64., columnStride * bytes) - bytes);
              double rowBlocks = std::ceil(double(g->m) / t[0]);
              scatteredBytes = rowBlocks * g->k * g->n * extra *
                               (g->grouped ? 1. : double(g->batches));
            }
          }
          CostEstimate e;
          e.scope = "full_call";
          e.bottleneck = "output block parallelism";
          e.terms["task_waves"] = {waves, "count"};
          e.terms["operand_bytes_per_worker"] = {reads, "B"};
          e.terms["vector_products"] = {ops, "count"};
          e.terms["shared_pack_bytes"] = {packBytes, "B"};
          e.terms["repeated_bf16_widen"] = {widenOps, "FLOP"};
          e.terms["page_walk_bytes"] = {translationBytes, "B"};
          e.terms["strided_column_bytes"] = {scatteredBytes, "B"};
          auto bw = intentdsl::quantity(ctx, "read_1", "B/ns"),
               rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
          if (bw) {
            if (t.size() == 6) {
              auto midBw = intentdsl::quantity(ctx, "read_1_mid_bw", "B/ns");
              auto syncNs = intentdsl::quantity(ctx, "sync", "ns");
              double tileBw = (packCode && midBw) ? *midBw : *bw;
              double prepSteps =
                  packCode == 3 ? (g->routed ? 1. : 2.) : (packCode ? 1. : 0.);
              double syncCost =
                  (syncNs && packBytes > 131072.) ? prepSteps * *syncNs : 0.;
              e.timeNs = (reads * workers) / tileBw +
                         (packBytes + translationBytes + scatteredBytes) / *bw +
                         syncCost +
                         (rate ? (ops * t[3] * 2. * workers +
                                  widenOps * workers) /
                                     *rate
                               : 0.);
            } else {
              e.timeNs = (reads * workers + packBytes + translationBytes +
                          scatteredBytes) /
                             *bw +
                         (rate ? (ops * t[3] + widenOps * workers) / *rate : 0);
            }
          }
          if (dot.getName().stripDialect() == "q4k_dot" && bw) {
            double bytes = waves * t[0] * (g->k / 256) * 144;
            auto dispatch = intentdsl::quantity(ctx, "launch", "ns");
            e.terms["record_bytes_per_worker"] = {bytes, "B"};
            e.timeNs =
                bytes * workers / *bw + (dispatch ? waves * *dispatch : 0.);
          }
          const std::string &name =
              t.size() == 6 ? caps->contracts[t[5]].estimator : caps->estimate;
          if (!name.empty()) {
            auto local = ctx.local(name);
            if (!local)
              return local.takeError();
            e.terms.insert(local->terms.begin(), local->terms.end());
            if (local->timeNs && e.timeNs)
              *e.timeNs += waves * *local->timeNs;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
