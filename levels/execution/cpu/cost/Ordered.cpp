#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "intentdsl/levels/execution/cpu/ir/ScoreCache.h"
#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
void ownerCost(Unit &u) {
  u.costModels.push_back(
      {{"ordered-tasks", u.qualify("ordered-owners"), "1",
        [](mlir::Operation *o, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto f = ctx.target.services.get<intentdsl::CallFacts>();
          auto n = intentdsl::kir::ownerCount(o, *f);
          auto cache = mlir::dyn_cast<mlir::DictionaryAttr>(c.value);
          auto w = mlir::cast<mlir::IntegerAttr>(cache ? cache.get("grain")
                                                       : c.value)
                       .getInt();
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
          if (auto panel = intentdsl::cpu::reusableScores(o, ctx.target)) {
            bool retained =
                cache &&
                (cache.get("scores") ||
                 (cache.get("window") &&
                  panel->keys <=
                      cache.getAs<mlir::IntegerAttr>("window").getInt()));
            double dots = double(n) * panel->keys * (retained ? 1 : 2);
            e.terms["score_dot_products"] = {dots, "count"};
            auto bandwidth = intentdsl::quantity(ctx, "read_1", "B/ns");
            if (e.timeNs && bandwidth && rate)
              *e.timeNs += 2. * dots * panel->depth / *rate +
                           (retained ? 8. * n * panel->keys / *bandwidth : 0.);
          }
          if (auto shared = intentdsl::cpu::sharedInterval(o, *f)) {
            bool together = cache && cache.get("inner");
            auto ownersPerLoad =
                together ? cache.getAs<mlir::IntegerAttr>("batch").getInt()
                         : int64_t(1);
            auto bytes = shared->readBytes * n / ownersPerLoad;
            // One of the two projections is replaced by a private FP32
            // store/load pair for the score of each owner.
            if (together && cache.get("window"))
              bytes = bytes * .5 + 8. * n;
            e.terms["common_input_bytes_per_member"] = {bytes, "B"};
            if (auto bw = intentdsl::quantity(ctx, "read_1", "B/ns");
                e.timeNs && bw)
              *e.timeNs += bytes / *bw;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
