#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace CI_UNIT_NAMESPACE {
void ownerSupplyCost(compilerinfra::Unit &u) {
  using namespace compilerinfra;
  u.costModels.push_back(
      {{"ordered-transfer", u.qualify("ordered-supply"), "2",
        [](mlir::Operation *o, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto fn = o->getParentOfType<mlir::func::FuncOp>();
          auto b = mlir::cast<mlir::DictionaryAttr>(c.value)
                       .getAs<mlir::IntegerAttr>("bytes")
                       .getInt();
          int reads = 0, writes = 0, slices = 0;
          int64_t foldSteps = 1;
          o->walk([&](mlir::Operation *v) {
            reads += mlir::isa<mlir::memref::LoadOp>(v);
            writes += mlir::isa<mlir::memref::StoreOp>(v);
            auto sn = v->getName().stripDialect();
            if (sn == "slice_read" || sn == "slice_write")
              ++slices;
            if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(v)) {
              if (auto dim = loop.getUpperBound()
                                 .getDefiningOp<mlir::memref::DimOp>()) {
                if (dim.getConstantIndex())
                  foldSteps = std::max<int64_t>(
                      foldSteps,
                      fn.getArgAttrOfType<mlir::DenseI64ArrayAttr>(
                          mlir::cast<mlir::BlockArgument>(dim.getSource())
                              .getArgNumber(),
                          "block.shape")[*dim.getConstantIndex()]);
              } else {
                mlir::Value iv = loop.getInductionVar();
                loop.walk([&](mlir::Operation *sub) {
                  if (sub->getName().stripDialect() == "slice_read" &&
                      mlir::isa<mlir::BlockArgument>(sub->getOperand(0))) {
                    auto coords = sub->getOperands().drop_front();
                    if (!coords.empty() && coords.back() == iv) {
                      auto sh = fn.getArgAttrOfType<mlir::DenseI64ArrayAttr>(
                          mlir::cast<mlir::BlockArgument>(sub->getOperand(0))
                              .getArgNumber(),
                          "block.shape");
                      int64_t rk =
                          mlir::cast<mlir::MemRefType>(sub->getOperand(0).getType())
                              .getRank();
                      if (sh && rk >= 2)
                        foldSteps = std::max<int64_t>(foldSteps, sh[rk - 2]);
                    }
                  }
                });
              }
            }
          });
          CostEstimate e;
          e.scope = "one owner";
          e.bottleneck = "state accesses";
          auto bw = intentdsl::quantity(ctx, "dma_in_bw", "B/ns"),
               lat = intentdsl::quantity(ctx, "dma_in_latency", "ns");
          if (auto plan = intentdsl::dsa::indexedRowAdd(fn, ctx.target)) {
            // Compare transfer issues and bytes on the same per-column basis.
            bool atomic = c.id == "indexed-row-add";
            double rows = plan->rows, width = plan->width;
            double issues =
                atomic ? (rows + 2. * std::ceil(rows / plan->batch)) / width
                       : rows * (reads + writes);
            double bytes = atomic ? rows * (12. + 4. / width) : rows * 16.;
            e.bottleneck =
                atomic ? "vector atomic updates" : "scalar indexed updates";
            e.terms["transfer_issues_per_owner"] = {issues, "count"};
            if (bw && lat)
              e.timeNs = issues * *lat + bytes / *bw;
            return e;
          }
          if (c.id == "panel-fold") {
            auto pf = intentdsl::dsa::panelFoldSupply(fn, ctx.target);
            double m = pf ? double(std::max<int64_t>(1, pf->rows)) : 64.0;
            double panels = std::max(1.0, std::ceil(foldSteps / 64.0));
            double transfers =
                pf ? double(pf->ownerTransferSites + pf->stripTransferSites) / m
                   : (2.0 + 3.0 * panels) / m;
            double bytesPerOwner =
                pf ? double(pf->ownerTransferBytes + pf->stripTransferBytes) / m
                   : double(b);
            e.terms["memory_access_sites"] = {double(reads + writes + slices),
                                              "count"};
            e.terms["private_slab_bytes"] = {double(b), "B"};
            if (bw && lat) {
              e.timeNs = transfers * *lat + bytesPerOwner / *bw;
              e.rows["0"] = {e.terms, e.timeNs};
            }
            return e;
          }
          auto streams = intentdsl::dsa::ownerStreams(fn, ctx.target);
          double transfers =
              b ? 2 : double(reads + writes) + double(slices) * foldSteps;
          double totalBytes = b ? 2.0 * b : 4.0 * transfers;
          if (!streams.transfers.empty()) {
            transfers = 0;
            for (auto t : streams.transfers)
              transfers += c.id == "streamed-owner-views"
                               ? 1.0 / std::max<int64_t>(1, streams.pack)
                               : t.steps;
            totalBytes = c.id == "streamed-owner-views"
                             ? double(streams.bytes) /
                                   std::max<int64_t>(1, streams.pack)
                             : double(streams.bytes);
          }
          e.terms["memory_access_sites"] = {double(reads + writes + slices),
                                            "count"};
          e.terms["private_slab_bytes"] = {double(b), "B"};
          if (bw && lat) {
            e.timeNs = transfers * *lat + totalBytes / *bw;
            e.rows["0"] = {e.terms, e.timeNs};
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
