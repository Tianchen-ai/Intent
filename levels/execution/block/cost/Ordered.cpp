#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "intentdsl/levels/execution/block/ir/ScorePanel.h"
#include "intentdsl/levels/execution/block/ir/SharedReads.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
          auto span = choice.getAs<mlir::IntegerAttr>("span");
          double panelBytes = 0;
          if (span) {
            auto lanesPerPanel = span.getInt() * lanes;
            // Independent contributions issue as a tile. Working sets wider
            // than a warp's register budget add register-pressure cost.
            dependency *=
                (1. + std::max(0., lanesPerPanel / 4096. - 1.)) / span.getInt();
            if (choice.getAs<mlir::StringAttr>("memory") == "matrix-fold") {
              issueGroups =
                  std::ceil(double(w) / 16.) * std::ceil(double(lanes) / 32.);
              double live = double(w) * lanes + double(span.getInt()) * lanes +
                            w * span.getInt();
              double bytesPerToken = 0;
              op->walk([&](mlir::Operation *load) {
                if (load->getName().stripDialect() != "slice_read" ||
                    !load->getParentOfType<mlir::scf::ForOp>())
                  return;
                auto ax = intentdsl::kir::memberAxis(load->getResult(0));
                bytesPerToken +=
                    (*f->shapeOf(ax->first))[ax->second]
                        .getAsInteger()
                        .value() *
                    mlir::cast<mlir::ShapedType>(load->getResult(0).getType())
                        .getElementTypeBitWidth() /
                    8.;
              });
              // The same K/V panel serves every owner row in this block.
              panelBytes = blocks * span.getInt() * bytesPerToken;
              dependency =
                  (1. + std::max(0., live / 32768. - 1.)) / span.getInt();
              e.terms["matrix_issue_groups"] = {issueGroups, "count"};
              e.terms["shared_panel_read_bytes"] = {panelBytes, "B"};
            }
          }
          if (auto panel = intentdsl::block::scorePanel(op, ctx.target)) {
            auto resident = choice.getAs<mlir::StringAttr>("memory") == "score";
            auto steps = resident ? 1. : 2. * panel->keys;
            dependency *= steps;
            if (resident)
              issueGroups =
                  std::ceil(w / 16.) * std::ceil(panel->keys / 64.) *
                  std::ceil(std::max(panel->depth, panel->features) / 32.);
            e.terms["sequential_key_steps"] = {steps, "count"};
          }
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
          double readCost = 0;
          if (auto r = intentdsl::block::sharedReads(op, *f)) {
            auto shared = choice.getAs<mlir::StringAttr>("memory") == "shared";
            for (auto *load : r->slices) {
              auto v = load->getResult(0);
              auto axis = intentdsl::kir::memberAxis(v);
              const auto *shape = f->shapeOf(axis->first);
              auto bytes = (*shape)[axis->second].getAsInteger().value() *
                           mlir::cast<mlir::ShapedType>(v.getType())
                               .getElementTypeBitWidth() /
                           8.;
              auto loop = load->getParentOfType<mlir::scf::ForOp>();
              auto bound =
                  loop.getUpperBound().getDefiningOp<mlir::memref::DimOp>();
              if (bound && bound.getConstantIndex())
                bytes *=
                    (*f->shapeOf(bound.getSource()))[*bound.getConstantIndex()]
                        .getAsInteger()
                        .value();
              readCost += bytes * (shared ? blocks : n);
            }
            e.terms["group_input_bytes"] = {readCost, "B"};
          }
          if (launch && sync)
            e.timeNs = *launch + *sync * (waves * issueGroups * dependency +
                                          blocks / std::max<double>(1, n));
          if (e.timeNs && readCost)
            if (auto rate = intentdsl::quantity(ctx, "read_1", "B/ns"))
              *e.timeNs += readCost / *rate;
          if (e.timeNs && panelBytes)
            if (auto rate = intentdsl::quantity(ctx, "read_1", "B/ns"))
              *e.timeNs += panelBytes / *rate / span.getInt();
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
