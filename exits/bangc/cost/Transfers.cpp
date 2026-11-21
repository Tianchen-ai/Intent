#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void dmaCost(Unit &u) {
  u.costModels.push_back(
      {{"dma", "levels/storage/dsa::supply", "1",
        [](Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          CostEstimate e;
          e.scope = "one row DMA";
          e.bottleneck = "transfer";
          auto [m, n] =
              ctx.target.services.get<intentdsl::CallFacts>()->geometry(op);
          double in = 0, out = 0, r = 0, w = 0, scatter = 0;
          op->walk([&](Operation *i) {
            auto k = i->getName().stripDialect();
            if (k != "load" && k != "store")
              return;
            double bytes = n *
                           cast<MemRefType>(i->getOperand(0).getType())
                               .getElementTypeBitWidth() /
                           8;
            if (k == "load") {
              auto choice = cast<DictionaryAttr>(c.value);
              double frequency = 1.;
              if (choice.getAs<BoolAttr>("persistent").getValue() &&
                  cast<MemRefType>(i->getOperand(0).getType()).getRank() == 1)
                frequency =
                    1. /
                    std::max(
                        1.,
                        std::ceil(m / choice.getAs<IntegerAttr>("participants")
                                          .getInt()));
              in += frequency * bytes;
              r += frequency;
            } else {
              out += bytes;
              ++w;
              if (auto axes =
                      i->getAttrOfType<DenseI64ArrayAttr>("block.member_axes");
                  axes && axes.size() == 2 && axes[0] == 1)
                scatter += bytes;
            }
          });
          auto bi = intentdsl::quantity(ctx, "dma_in_bw", "B/ns"),
               bo = intentdsl::quantity(ctx, "dma_out_bw", "B/ns"),
               li = intentdsl::quantity(ctx, "dma_in_latency", "ns"),
               lo = intentdsl::quantity(ctx, "dma_out_latency", "ns"),
               bw = intentdsl::quantity(ctx, "write_16", "B/ns");
          e.terms = {{"incoming", {in, "B"}},
                     {"outgoing", {out, "B"}},
                     {"requests", {r + w, "count"}}};
          if (bi && bo && li && lo && bw) {
            double fetch = in / *bi + r * *li,
                   drain = (out - scatter) / *bo + scatter / *bw + w * *lo;
            e.terms["fetch"] = {fetch, "ns"};
            e.terms["writeback"] = {drain, "ns"};
            e.timeNs = fetch + drain;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
