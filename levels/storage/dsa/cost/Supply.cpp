#include "intentdsl/levels/execution/block/ir/Traffic.h"
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
          auto [keys,segmentWork]=intentdsl::block::realizedTraffic(op,*facts);
          auto depth = d.getAs<IntegerAttr>("depth").getInt();
          double participants = d.getAs<IntegerAttr>("participants").getInt();
          double batch = d.getAs<IntegerAttr>("batch")
                             ? d.getAs<IntegerAttr>("batch").getInt()
                             : 1.;
          auto configs = op->getAttrOfType<ArrayAttr>("block.configs");
          unsigned ops = 0;
          unsigned summaries = 0, scans = 0, atomics=0;
          op->walk([&](Operation *i) {
            auto k = i->getName().stripDialect();
            ops += k == "mul" || k == "add" || k == "reduce" || k == "exp" ||
                   k == "rsqrt" || k == "div";
            summaries += k == "summary_reduce";
            scans += k == "scan_sum";
            atomics += k == "atomic_add_at";
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
            double effGroup = group * batch;
            double steps = std::ceil(std::ceil(m / effGroup) / participants);
            double tileFraction = 1.;
            if(d.contains("independent_columns")){
              auto width=cast<DictionaryAttr>(*it).getAs<IntegerAttr>("width").getInt();
              steps=std::ceil(std::ceil(m/effGroup)*std::ceil(n/width)/participants);
              tileFraction = std::min(1., double(width) / std::max(1., double(n)));
            }
            RowPrediction pred;
            pred.terms = dma->terms;
            pred.terms["steps"] = {steps, "count"};
            if(segmentWork)pred.terms["packed_member_reads"]={segmentWork,"B"};
            if(keys)pred.terms["key_input_bytes"]={keys,"B"};
            double merges = effGroup * std::max(0., n - 1.) * summaries;
            pred.terms["summary_merges_per_group"] = {merges, "count"};
            pred.terms["serial_prefix_adds"]={effGroup*std::max(0.,n-1.)*scans,"count"};
            if(atomics)pred.terms["atomic_updates_per_group"]={effGroup*n*atomics,"count"};
            pred.terms["working_set"] = {
                double(row.getAs<IntegerAttr>("bytes").getInt()), "B"};
            if (dma->timeNs && launch && sync && vector && startup) {
              double fetch = effGroup * tileFraction * e.terms.at("fetch").value,
                     compute = tileFraction * ((effGroup * n * std::max(0.,ops-e.terms.at("native_operations").value) + 8 * merges + pred.terms["serial_prefix_adds"].value) / *vector + effGroup * e.terms.at("native_compute").value),
                     write = effGroup * tileFraction * e.terms.at("writeback").value;
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
              if(atomics)pred.timeNs=*pred.timeNs+steps*effGroup*n*atomics*(*sync);
              if(segmentWork || keys){
                auto rate=intentdsl::quantity(ctx,"read_1","B/ns");
                if(rate)*pred.timeNs+=(segmentWork+keys)/ *rate;
              }
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
