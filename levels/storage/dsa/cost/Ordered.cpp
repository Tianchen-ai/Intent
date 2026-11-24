#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace CI_UNIT_NAMESPACE {
void ownerSupplyCost(compilerinfra::Unit &u) {
  using namespace compilerinfra;
  u.costModels.push_back(
      {{"ordered-transfer", u.qualify("ordered-supply"), "1",
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
            if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(v))
              if (auto dim = loop.getUpperBound()
                                 .getDefiningOp<mlir::memref::DimOp>())
                if (dim.getConstantIndex())
                  foldSteps = std::max<int64_t>(
                      foldSteps,
                      fn.getArgAttrOfType<mlir::DenseI64ArrayAttr>(
                          mlir::cast<mlir::BlockArgument>(dim.getSource())
                              .getArgNumber(),
                          "block.shape")[*dim.getConstantIndex()]);
          });
          CostEstimate e;
          e.scope = "one owner";
          e.bottleneck = "state accesses";
          auto bw = intentdsl::quantity(ctx, "dma_in_bw", "B/ns"),
               lat = intentdsl::quantity(ctx, "dma_in_latency", "ns");
          if (c.id == "panel-fold") {
            auto pf = intentdsl::dsa::panelFoldSupply(fn, ctx.target);
            double m = pf ? pf->rows : 64.0;
            double panels = std::max(1.0, std::ceil(foldSteps / 64.0));
            double transfers = (2.0 + 3.0 * panels) / m;
            double bytesPerOwner =
                pf ? double(pf->depth + pf->features +
                            3.0 * foldSteps * (pf->depth + pf->features) / m) *
                         2.0
                   : double(b);
            e.terms["memory_access_sites"] = {double(reads + writes + slices),
                                              "count"};
            e.terms["private_slab_bytes"] = {double(b), "B"};
            if (bw && lat)
              e.timeNs = transfers * *lat + bytesPerOwner / *bw;
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
          if (bw && lat)
            e.timeNs = transfers * *lat + totalBytes / *bw;
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
