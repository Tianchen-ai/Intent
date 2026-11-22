#include "intentdsl/levels/execution/cpu/ir/Owners.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
void ownerCost(Unit &u) {
  u.costModels.push_back(
      {{"ordered-tasks", u.qualify("ordered-owners"), "1",
        [](mlir::Operation *o, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto f = ctx.target.services.get<intentdsl::CallFacts>();
          auto n = intentdsl::kir::ownerCount(o, *f);
          auto w = mlir::cast<mlir::IntegerAttr>(c.value).getInt();
          auto workers =
              ctx.target.services.get<intentdsl::cpu::RowMachine>()->workers;
          double tasks = llvm::divideCeil(n, w),
                 waves = std::ceil(tasks / workers);
          CostEstimate e;
          e.scope = "full_call";
          e.bottleneck = "owner task granularity";
          e.terms["native_tasks"] = {tasks, "count"};
          e.terms["worker_waves"] = {waves, "count"};
          auto launch = intentdsl::quantity(ctx, "launch", "ns"),
               rate = intentdsl::quantity(ctx, "vector", "FLOP/ns");
          if (launch && rate)
            e.timeNs = *launch +
                       waves * (1 + std::min(w, n)) * workers / *rate +
                       tasks / *rate;
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
