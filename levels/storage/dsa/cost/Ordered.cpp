#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
namespace CI_UNIT_NAMESPACE {
void ownerSupplyCost(compilerinfra::Unit &u) {
  using namespace compilerinfra;
  u.costModels.push_back(
      {{"ordered-transfer", u.qualify("ordered-supply"), "1",
        [](mlir::Operation *o, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto b = mlir::cast<mlir::DictionaryAttr>(c.value)
                       .getAs<mlir::IntegerAttr>("bytes")
                       .getInt();
          int reads = 0, writes = 0;
          o->walk([&](mlir::Operation *v) {
            reads += mlir::isa<mlir::memref::LoadOp>(v);
            writes += mlir::isa<mlir::memref::StoreOp>(v);
          });
          CostEstimate e;
          e.scope = "one owner";
          e.bottleneck = "state accesses";
          auto bw = intentdsl::quantity(ctx, "dma_in_bw", "B/ns"),
               lat = intentdsl::quantity(ctx, "dma_in_latency", "ns");
          auto streams = intentdsl::dsa::ownerStreams(
              o->getParentOfType<mlir::func::FuncOp>(), ctx.target);
          double transfers = b ? 2 : reads + writes;
          if (!streams.transfers.empty()) {
            transfers = 0;
            for (auto t : streams.transfers)
              transfers += c.id == "streamed-owner-views" ? 1 : t.steps;
          }
          e.terms["memory_access_sites"] = {double(reads + writes), "count"};
          e.terms["private_slab_bytes"] = {double(b), "B"};
          if (bw && lat)
            e.timeNs = transfers * *lat + (streams.bytes ? streams.bytes
                                           : b           ? 2 * b
                                                         : 4 * transfers) /
                                              *bw;
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
