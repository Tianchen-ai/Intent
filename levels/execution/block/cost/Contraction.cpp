#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
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
          bool records = false;
          double recordBytes = 144.0, recordElems = 256.0;
          const intentdsl::kir::QuantFormatSpec *qSpec = nullptr;
          for (mlir::Operation &o : op->getRegion(0).front()) {
            auto k = o.getName().stripDialect();
            if (k == "q4k_dot") {
              records = true;
            } else if (k == "quant_contract") {
              qSpec = intentdsl::kir::quantFormatSpec(o);
              if (qSpec &&
                  qSpec->roleLayout == intentdsl::kir::QuantRoleLayout::RecordGemv) {
                records = true;
                recordBytes = qSpec->recordBytes;
                recordElems = qSpec->blockElements;
              }
            }
          }
          double jobs = shape->batches * std::ceil(shape->m / tm) *
                        std::ceil(shape->n / tn);
          double panels = records ? std::ceil(shape->k / recordElems)
                                  : std::ceil(shape->k / tk);
          double panelBytes =
              records ? (tm * recordBytes + recordElems * 4.0)
              : (qSpec && qSpec->roleLayout ==
                              intentdsl::kir::QuantRoleLayout::BlockScaledPair)
                  ? ((tm + tn) * tk * (qSpec->packBits / 8.0 +
                                       1.0 / std::max(1, qSpec->blockElements)))
              : (qSpec && qSpec->roleLayout ==
                              intentdsl::kir::QuantRoleLayout::GroupedWeightRhs)
                  ? (tm * tk * 2.0 +
                     tn * tk * (qSpec->packBits / 8.0 +
                                2.5 / std::max(1, qSpec->blockElements)))
                  : ((tm + tn) * tk * 2.0);
          double mBlocks = shape->batches * std::ceil(shape->m / tm);
          double read =
              records ? (panels * (shape->batches * shape->m * recordBytes +
                                   mBlocks * recordElems * 4.0))
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
          // number of operand bytes unchanged; record dots amortize activation
          // quantization collectives across tm rows per CTA panel.
          double issues = records ? 0.0 : (2 * waves * panels);
          e.terms["panel_load_issues"] = {issues, "count"};
          auto sync = intentdsl::quantity(ctx, "sync", "ns");
          double waveBytes =
              records ? (tm * recordBytes *
                             std::max<int64_t>(1, f->participants) +
                         recordElems * 4.0)
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
