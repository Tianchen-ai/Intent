#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void supplyCost(Unit &u) {
  u.costModels.push_back(
      {{"pipeline-time", u.qualify("supply"), "1",
        [](Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto *facts = ctx.target.services.get<intentdsl::CallFacts>();
          auto [m, n] = facts->geometry(op);
          auto dma = ctx.local(facts->localEstimator);
          if (!dma)
            return dma.takeError();
          auto e = *dma;
          e.scope = "full_call";
          e.bottleneck = "supply/compute";
          auto sync = intentdsl::quantity(ctx, "sync", "ns"),
               launch = intentdsl::quantity(ctx, "launch", "ns"),
               vector = intentdsl::quantity(ctx, "compute", "lane/ns");
          auto startup = intentdsl::quantity(ctx, "first_io", "ns");
          auto d = cast<DictionaryAttr>(c.value);
          auto depth = d.getAs<IntegerAttr>("depth").getInt();
          double participants = d.getAs<IntegerAttr>("participants").getInt();
          auto configs = op->getAttrOfType<ArrayAttr>("block.configs");
          unsigned ops = 0;
          op->walk([&](Operation *i) {
            auto k = i->getName().stripDialect();
            ops += k == "mul" || k == "add" || k == "reduce" || k == "exp" ||
                   k == "rsqrt" || k == "div";
          });
          e.timeNs.reset();
          for (auto a : d.getAs<ArrayAttr>("rows")) {
            auto row = cast<DictionaryAttr>(a);
            auto id = row.get("id");
            auto it = llvm::find_if(configs, [&](Attribute v) {
              return cast<DictionaryAttr>(v).get("id") == id;
            });
            double group =
                cast<DictionaryAttr>(*it).getAs<IntegerAttr>("rows").getInt();
            double steps = std::ceil(std::ceil(m / group) / participants);
            RowPrediction pred;
            pred.terms = dma->terms;
            pred.terms["steps"] = {steps, "count"};
            pred.terms["working_set"] = {
                double(row.getAs<IntegerAttr>("bytes").getInt()), "B"};
            if (dma->timeNs && launch && sync && vector && startup) {
              double fetch = group * e.terms.at("fetch").value,
                     compute = group * (n * std::max(0.,ops-e.terms.at("native_operations").value) / *vector + e.terms.at("native_compute").value),
                     write = group * e.terms.at("writeback").value;
              double overlap = depth == 2 ? std::max(0., steps - 1) *
                                                std::min(fetch, compute)
                                          : 0.;
              pred.terms["overlapped"] = {overlap, "ns"};
              pred.terms["fill"] = {fetch, "ns"};
              pred.terms["drain"] = {compute + write, "ns"};
              pred.terms["launch"] = {*launch, "ns"};
              pred.terms["first_io_excess"] = {*startup, "ns"};
              pred.terms["wait"] = {depth == 2 ? steps * *sync : 0., "ns"};
              pred.timeNs = *launch + (steps ? *startup : 0.) + steps * (fetch + compute + write) -
                            overlap + (depth == 2 ? steps * *sync : 0.);
              if (!e.timeNs || *pred.timeNs < *e.timeNs)
                e.timeNs = pred.timeNs;
            }
            e.rows[std::to_string(cast<IntegerAttr>(id).getInt())] =
                std::move(pred);
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
