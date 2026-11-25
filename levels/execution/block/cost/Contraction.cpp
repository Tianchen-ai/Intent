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
          bool isSplit = tile.size() == 4 && (tile[3] == 1 || tile[3] == 2);
          bool isSplitLib = tile.size() == 4 && tile[3] == 2;
          double tm = tile[0], tn = tile[1], tk = tile[2];
          bool records = false;
          double recordBytes = 144.0, recordElems = 256.0;
          double lhsElemBytes = 2.0, outElemBytes = 2.0;
          const intentdsl::kir::QuantFormatSpec *qSpec = nullptr;
          for (mlir::Operation &o : op->getRegion(0).front()) {
            auto k = o.getName().stripDialect();
            if (k == "q4k_dot") {
              records = true;
            } else if (k == "quant_contract") {
              qSpec = intentdsl::kir::quantFormatSpec(o);
              if (auto mem = mlir::dyn_cast<mlir::MemRefType>(
                      o.getOperand(0).getType()))
                lhsElemBytes =
                    llvm::divideCeil(mem.getElementTypeBitWidth(), 8u);
              if (auto omem = mlir::dyn_cast<mlir::MemRefType>(
                      op->getOperand(0).getType()))
                outElemBytes =
                    llvm::divideCeil(omem.getElementTypeBitWidth(), 8u);
              if (qSpec &&
                  qSpec->roleLayout == intentdsl::kir::QuantRoleLayout::RecordGemv) {
                records = true;
                recordBytes = qSpec->recordBytes;
                recordElems = qSpec->blockElements;
              }
            }
          }
          bool groupedRhs =
              qSpec && qSpec->roleLayout ==
                           intentdsl::kir::QuantRoleLayout::GroupedWeightRhs;
          double groupedRhsB =
              groupedRhs ? (qSpec->packBits / 8.0 +
                            2.5 / std::max(1, qSpec->blockElements))
                         : 0.0;
          auto mt = ctx.target.services.get<intentdsl::block::MatrixTarget>();
          auto bw = intentdsl::quantity(ctx, "read_1", "B/ns"),
               wbw = intentdsl::quantity(ctx, "write_1", "B/ns"),
               sbw = intentdsl::quantity(ctx, "read_1_shared_bw", "B/ns"),
               swbw = intentdsl::quantity(ctx, "write_1_shared_bw", "B/ns"),
               startup = intentdsl::quantity(ctx, "launch", "ns");
          if (groupedRhs && mt && mt->supportsSplitGroupedRhs && bw && startup) {
            double dramR = *bw;
            double dramW = wbw.value_or(dramR);
            double l2R = std::max(sbw.value_or(dramR), dramR);
            double l2W = std::max(swbw.value_or(dramW), dramW);
            double sms = std::max<double>(1.0, f->participants);
            double B_packed = double(shape->k) * double(shape->n) * groupedRhsB;
            double B_unpacked =
                double(shape->k) * double(shape->n) * lhsElemBytes;
            double B_lhs = double(shape->m) * double(shape->k) * lhsElemBytes;
            double B_out = double(shape->m) * double(shape->n) * outElemBytes;
            double flops =
                2.0 * double(shape->m) * double(shape->n) * double(shape->k);
            double smMma = mt->smMmaOpsPerNs > 0.0 ? mt->smMmaOpsPerNs : 2000.0;
            CostEstimate e;
            e.scope = "full_call";
            if (isSplit) {
              bool fitsL2 = mt->l2CacheBytes > 0 &&
                            B_unpacked <= 0.85 * double(mt->l2CacheBytes);
              double stage1WriteBw = fitsL2 ? l2W : dramW;
              double dqWriteScale =
                  mt->splitDequantWriteScale > 0.0 ? mt->splitDequantWriteScale
                                                   : 1.0;
              double dqNs = *startup + B_packed / dramR +
                            B_unpacked * dqWriteScale / stage1WriteBw;
              double gemmNs = 0.0;
              double jobs = 0.0, panels = 0.0, waves = 1.0;
              if (isSplitLib) {
                double libMma = mt->librarySmMmaOpsPerNs > 0.0
                                    ? mt->librarySmMmaOpsPerNs
                                    : smMma * 1.15;
                double sDram = (B_lhs + (fitsL2 ? 0.0 : B_unpacked)) / dramR +
                               B_out / dramW;
                double sL2 = (B_unpacked + B_lhs) / l2R;
                double sMma = flops / (sms * libMma);
                gemmNs = *startup + sDram + std::max(sL2, sMma) +
                         0.25 * std::min(sL2, sMma);
                e.bottleneck = "split dequant + library contraction";
              } else {
                double mBlocks = std::ceil(shape->m / tm);
                double nBlocks = std::ceil(shape->n / tn);
                jobs = mBlocks * nBlocks;
                panels = std::ceil(shape->k / tk);
                waves = std::ceil(jobs / sms);
                double activeSms = std::min(jobs, sms);
                double sDram = (B_lhs + (fitsL2 ? 0.0 : B_unpacked)) / dramR +
                               B_out / dramW;
                double sL2 =
                    (B_unpacked * (1.0 + 0.25 * std::max(0.0, mBlocks - 1.0)) +
                     B_lhs) /
                    l2R;
                double sMma = flops / (activeSms * smMma);
                // Prefer tiles that fill SMs and use deeper K-panels cleanly.
                double waveBias = waves * panels * 16.0;
                gemmNs = *startup + sDram + std::max(sL2, sMma) +
                         0.35 * std::min(sL2, sMma) + waveBias;
                e.bottleneck = "split dequant + dense contraction";
              }
              e.terms["dequant_stage_ns"] = {dqNs, "ns"};
              e.terms["gemm_stage_ns"] = {gemmNs, "ns"};
              e.terms["workspace_bytes"] = {B_unpacked, "B"};
              e.terms["output_tiles"] = {jobs, "count"};
              e.timeNs = dqNs + gemmNs;
              return e;
            }
            // Fused grouped-RHS contraction on the same physical nanosecond scale.
            double mBlocks = shape->batches * std::ceil(shape->m / tm);
            double nBlocks = std::ceil(shape->n / tn);
            double jobs = mBlocks * nBlocks;
            double panels = std::ceil(shape->k / tk);
            double waves = std::ceil(jobs / sms);
            double activeSms = std::min(jobs, sms);
            double fMem = (B_lhs + B_packed * mBlocks) / dramR + B_out / dramW;
            double fMma = flops / (activeSms * smMma);
            // Each M-block wave decodes all K*N weights inside the MMA loop.
            double fDequant = (B_unpacked * mBlocks) / (0.45 * l2R);
            double waveBias = waves * panels * 16.0;
            e.bottleneck = "fused grouped-RHS contraction";
            e.terms["operand_bytes"] = {B_lhs + B_packed * mBlocks, "B"};
            e.terms["output_tiles"] = {jobs, "count"};
            e.terms["matrix_panel_waves"] = {waves * panels, "count"};
            e.timeNs = *startup + fMem + std::max(fMma, fDequant) +
                       0.5 * std::min(fMma, fDequant) + waveBias;
            return e;
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
              : groupedRhs
                  ? (tm * tk * 2.0 + tn * tk * groupedRhsB)
                  : ((tm + tn) * tk * 2.0);
          double mBlocks = shape->batches * std::ceil(shape->m / tm);
          double read =
              records ? (panels * (shape->batches * shape->m * recordBytes +
                                   mBlocks * recordElems * 4.0))
              : groupedRhs
                  ? (shape->batches *
                     (shape->m * shape->k * 2.0 +
                      shape->n * shape->k * groupedRhsB * mBlocks))
                  : (jobs * panels * panelBytes);
          double waves =
              std::ceil(jobs / std::max<int64_t>(1, f->participants));
          CostEstimate e;
          e.scope = "full_call";
          e.bottleneck = "matrix tile supply";
          e.terms["operand_bytes"] = {read, "B"};
          e.terms["output_tiles"] = {jobs, "count"};
          e.terms["matrix_panel_waves"] = {waves * panels, "count"};
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
          if (bw && startup) {
            if (groupedRhs) {
              double l2Bw = sbw.value_or(*bw * 4.0);
              double l2Traffic = jobs * panels * panelBytes;
              double activeSms = std::min<double>(
                  jobs, std::max<int64_t>(1, f->participants));
              double smUnderutil =
                  std::max<double>(1, f->participants) / std::max(1.0, activeSms);
              e.timeNs = *startup + read / *bw + l2Traffic / l2Bw +
                         (sync ? issues * *sync * smUnderutil : 0);
            } else {
              e.timeNs = *startup + read / *bw +
                         waves * panels * waveBytes / *bw +
                         (sync ? issues * *sync : 0);
            }
          }
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
