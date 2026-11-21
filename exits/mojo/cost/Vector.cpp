#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void vectorCost(Unit &u) {
  for (bool permuted : {false, true})
    u.costModels.push_back(
        {{permuted ? "vector-permutation" : "vector",
          permuted ? "levels/execution/cpu::permutation"
                   : "levels/execution/cpu::partition",
          "1",
          [](Operation *op, const StructuralCandidate &c,
             CostContext &ctx) -> llvm::Expected<CostEstimate> {
            auto [m, n] =
                ctx.target.services.get<intentdsl::CallFacts>()->geometry(op);
            auto d = cast<DictionaryAttr>(c.value);
            if (d.getAs<StringAttr>("walk") == "destination")
              std::swap(m, n);
            auto lanes = ctx.target.services.get<intentdsl::cpu::RowMachine>()
                             ->vectorBits /
                         32;
            unsigned ops = 0;
            op->walk([&](Operation *i) {
              auto k = i->getName().stripDialect();
              ops += k == "add" || k == "mul" || k == "exp" || k == "sum" ||
                     k == "max_num";
            });
            CostEstimate e;
            e.scope = "vector loop";
            e.bottleneck = "vector issue";
            double issued = m * std::ceil(n / lanes) * lanes;
            e.terms["padded_lanes"] = {issued - m * n, "lane"};
            e.terms["arithmetic"] = {issued * ops, "FLOP"};
            auto rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
            if (rate)
              e.timeNs = issued * ops / *rate;
            return e;
          }},
         COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
