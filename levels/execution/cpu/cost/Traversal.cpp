#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/algorithm/kir/ir/Permutation.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerCost(Unit &u) {
  for (bool permuted : {false, true})
    u.costModels.push_back(
        {{permuted ? "permutation-time" : "task-time",
          u.qualify(permuted ? "permutation" : "partition"), "1",
          [](Operation *op, const StructuralCandidate &c,
             CostContext &ctx) -> llvm::Expected<CostEstimate> {
            auto *facts = ctx.target.services.get<intentdsl::CallFacts>();
            auto [m, n] = facts->geometry(op);
            auto d = cast<DictionaryAttr>(c.value);
            auto walk = d.getAs<StringAttr>("walk");
            auto rows = d.getAs<ArrayAttr>("rows");
            auto local = ctx.local(
                facts->localEstimator +
                (intentdsl::rectangularCopy(op) ? "-permutation" : ""));
            if (!local)
              return local.takeError();
            CostEstimate e = *local;
            e.scope = "full_call";
            e.bottleneck = "memory traffic";
            auto rd = intentdsl::quantity(ctx, "read_1", "B/ns"),
                 wr = intentdsl::quantity(ctx, "write_1", "B/ns");
            auto rs = intentdsl::quantity(ctx, "read_16", "B/ns"),
                 ws = intentdsl::quantity(ctx, "write_16", "B/ns");
            auto launch = intentdsl::quantity(ctx, "launch", "ns"),
                 sync = intentdsl::quantity(ctx, "sync", "ns");
            double read = 0, write = 0, readNs = 0, writeNs = 0, lines = 0;
            llvm::Error strideError=llvm::Error::success();
            op->walk([&](Operation *item) {
              if(strideError)return;
              auto k = item->getName().stripDialect();
              if (k != "read" && k != "write")
                return;
              auto arg = dyn_cast<BlockArgument>(item->getOperand(0));
              if (!arg)
                return;
              auto v = cast<MemRefType>(arg.getType());
              auto bytes = v.getElementTypeBitWidth() / 8;
              bool reversed = item->hasAttr("axes") != (walk == "destination");
              auto axis = v.getRank() == 2 && reversed ? 0 : v.getRank() - 1;
              auto stride=facts->axisStride(item,arg.getArgNumber(),axis);
              if(!stride){strideError=stride.takeError();return;}
              double step=std::abs(*stride);
              double mix = std::clamp((step - 1) / 15., 0., 1.);
              if (walk == "tile")
                mix *=
                    1. / std::max(1., std::min(32., n) * bytes / sizeof(float));
              double traffic = m * n * bytes;
              lines +=
                  std::ceil(traffic * std::min(std::max(step, 1.), 16.) / 64.);
              if (k == "read") {
                read += traffic;
                if (rd && rs)
                  readNs += traffic * ((1 - mix) / *rd + mix / *rs);
              } else {
                write += traffic;
                if (wr && ws)
                  writeNs += traffic * ((1 - mix) / *wr + mix / *ws);
              }
            });
            if(strideError)return std::move(strideError);
            e.terms["read"] = {read, "B"};
            e.terms["write"] = {write, "B"};
            e.terms["line_touches"] = {lines, "line"};
            auto arithmetic = e.timeNs;
            e.timeNs.reset();
            for (auto [i, row] : llvm::enumerate(rows)) {
              auto grain = cast<DictionaryAttr>(row)
                               .getAs<IntegerAttr>("grain")
                               .getInt();
              double groups =
                  walk == "tile"
                      ? std::ceil(n / 32.) * std::ceil(m / 32.)
                      : std::ceil((walk == "destination" ? n : m) / grain);
              double active = std::min(groups, double(facts->participants));
              RowPrediction pred;
              pred.terms = e.terms;
              pred.terms["tasks"] = {groups, "count"};
              pred.terms["active_workers"] = {active, "count"};
              if (rd && wr && rs && ws && launch && sync && arithmetic) {
                double work = std::max(readNs + writeNs, *arithmetic);
                pred.timeNs =
                    *launch + *sync +
                    work * (groups ? std::ceil(groups / facts->participants) *
                                         facts->participants / groups
                                   : 0.);
                if (!e.timeNs || *pred.timeNs < *e.timeNs)
                  e.timeNs = pred.timeNs;
              }
              e.rows[std::to_string(i)] = std::move(pred);
            }
            return e;
          }},
         COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
