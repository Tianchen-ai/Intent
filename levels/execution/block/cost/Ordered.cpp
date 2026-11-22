#include "intentdsl/levels/execution/block/ir/Owners.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
void orderedCost(Unit &u) {
  u.costModels.push_back(
      {{"owner-issue", u.qualify("ordered-owners"), "1",
        [](mlir::Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto f = ctx.target.services.get<intentdsl::CallFacts>();
          auto n = intentdsl::block::ownerCount(op, *f);
          auto choice = mlir::cast<mlir::DictionaryAttr>(c.value);
          auto w = choice.getAs<mlir::IntegerAttr>("width").getInt();
          auto caps = ctx.target.services.get<intentdsl::block::OwnerMachine>();
          int64_t lanes = 1;
          op->walk([&](mlir::Operation *inst) {
            for (auto v : inst->getResults())
              if (auto axis = intentdsl::kir::memberAxis(v)) {
                if (auto *sh = f->shapeOf(axis->first))
                  lanes = std::max(
                      lanes, (*sh)[axis->second].getAsInteger().value_or(1));
              }
          });
          auto issue = caps ? caps->parallelMembers : 1;
          double blocks = llvm::divideCeil(n, w),
                 waves =
                     std::ceil(blocks / std::max<int64_t>(1, f->participants));
          CostEstimate e;
          e.scope = "full_call";
          e.bottleneck = "sequential work per owner";
          e.terms["independent_blocks"] = {blocks, "count"};
          e.terms["issue_waves"] = {waves, "count"};
          auto launch = intentdsl::quantity(ctx, "launch", "ns"),
               sync = intentdsl::quantity(ctx, "sync", "ns");
          double issueGroups = std::ceil(double(w) * lanes / issue);
          double dependency = 1;
          if (intentdsl::block::canForwardOwners(op, ctx.target)) {
            auto sh = f->shapeOf(op->getOperand(0));
            auto n = (*sh)[sh->size() - 1].getAsInteger().value_or(1);
            // Register forwarding keeps scalar update chains, whereas the
            // rank-one schedule updates all residual columns in parallel.
            auto mode = choice.getAs<mlir::StringAttr>("memory");
            dependency = mode == "columns"   ? 1.
                         : mode == "forward" ? n
                                             : double(n) * n;
          }
          e.terms["member_issue_groups"] = {issueGroups, "count"};
          if (intentdsl::block::canGroupReads(op, ctx.target, 4)) {
            auto fetch = choice.getAs<mlir::IntegerAttr>("prefetch");
            double span = fetch ? fetch.getInt() : 1;
            double independentLoads = 0;
            op->walk([&](mlir::Operation *v) {
              if (v->getName().stripDialect() == "slice_read" ||
                  v->getName().getStringRef() == "memref.load")
                ++independentLoads;
            });
            double live = span * w * lanes * independentLoads;
            dependency = (1 + independentLoads / span) *
                         (1 + std::max(0., live - 4096) / 4096);
            e.terms["grouped_read_elements"] = {live, "count"};
          }
          if (launch && sync)
            e.timeNs = *launch + *sync * (waves * issueGroups * dependency +
                                          blocks / std::max<double>(1, n));
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
