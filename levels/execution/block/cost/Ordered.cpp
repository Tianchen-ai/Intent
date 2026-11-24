#include "intentdsl/levels/execution/block/ir/FoldIssue.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
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
          // A slice indexed by the fold induction variable is loaded once per
          // visited token per tile of owners. Use its actual access dimension,
          // not the number of tokens in a single issue packet.
          double sweepBytesPerOwner = 0;
          for (auto loop :
               op->getRegion(0).front().getOps<mlir::scf::ForOp>()) {
            if (!loop->hasAttr("kir.reassociate"))
              continue;
            double tokenBytes = 0;
            int64_t extent = 0;
            for (auto &inst : loop.getBody()->without_terminator()) {
              bool slice = inst.getName().stripDialect() == "slice_read";
              if (!slice && !mlir::isa<mlir::memref::LoadOp>(inst))
                continue;
              auto sh = f->shapeOf(inst.getOperand(0));
              if (!sh)
                continue;
              unsigned at = 0;
              for (mlir::Value index : inst.getOperands().drop_front()) {
                if (slice &&
                    at == static_cast<unsigned>(
                              inst.getAttrOfType<mlir::IntegerAttr>("axis")
                                  .getInt()))
                  ++at;
                if (index == loop.getInductionVar()) {
                  int64_t depth = 1;
                  if (slice)
                    depth = (*sh)[inst.getAttrOfType<mlir::IntegerAttr>("axis")
                                      .getInt()]
                                .getAsInteger()
                                .value_or(1);
                  tokenBytes +=
                      depth *
                      mlir::cast<mlir::MemRefType>(inst.getOperand(0).getType())
                          .getElementTypeBitWidth() /
                      8.;
                  extent =
                      std::max(extent, (*sh)[at].getAsInteger().value_or(0));
                }
                ++at;
              }
            }
            if (!extent)
              continue;
            // For a causal one-to-one owner axis the mean prefix or suffix
            // length is (L+1)/2. Otherwise count the full sweep.
            double visits = extent;
            auto iv = mlir::dyn_cast<mlir::BlockArgument>(loop.getLowerBound());
            if (!iv)
              iv = mlir::dyn_cast_if_present<mlir::BlockArgument>(
                  intentdsl::block::causalPrefixOwner(loop));
            if (iv && iv.getOwner() == loop->getBlock()) {
              auto axes = op->getAttrOfType<mlir::DenseI64ArrayAttr>("axes");
              auto sh = f->shapeOf(op->getOperand(iv.getArgNumber()));
              if (axes && sh &&
                  (*sh)[axes[iv.getArgNumber()]].getAsInteger() == extent)
                visits = (extent + 1.) / 2;
            }
            sweepBytesPerOwner += visits * tokenBytes;
          }
          if (sweepBytesPerOwner)
            e.terms["sweep_bytes_per_owner"] = {sweepBytesPerOwner, "B"};
          bool isSharedMode = choice.getAs<mlir::StringAttr>("memory") == "shared";
          bool isMatrixFold =
              choice.getAs<mlir::StringAttr>("memory") == "matrix-fold";
          double panelBytes =
              sweepBytesPerOwner * ((isSharedMode || isMatrixFold) ? blocks : n),
                 spillBytes = 0;
          if (span) {
            auto lanesPerPanel = span.getInt() * lanes;
            // Independent contributions issue as a tile. Working sets wider
            // than a warp's register budget add register-pressure cost.
            dependency *=
                (1. + std::max(0., lanesPerPanel / 4096. - 1.)) / span.getInt();
            if (choice.getAs<mlir::StringAttr>("memory") == "matrix-fold") {
              issueGroups =
                  std::ceil(double(w) / 16.) * std::ceil(double(lanes) / 32.);
              double physicalRows = std::max<int64_t>(w, 16);
              double live = physicalRows * lanes +
                            double(span.getInt()) * lanes +
                            physicalRows * span.getInt();
              double effectiveBlocks = std::max<double>(
                  blocks, std::max<int64_t>(1, f->participants));
              panelBytes = effectiveBlocks * sweepBytesPerOwner;
              dependency =
                  (1. + std::max(0., live / 32768. - 1.)) / span.getInt();
              // Independent matrix updates compete for resident accumulator
              // fragments and exchange score columns between issue groups.
              unsigned carries = 0, projections = 0;
              double operandWords = 1.;
              for (auto loop :
                   op->getRegion(0).front().getOps<mlir::scf::ForOp>()) {
                if (!loop->hasAttr("kir.reassociate"))
                  continue;
                for (auto &inst : loop.getBody()->without_terminator())
                  if (inst.getName().stripDialect() == "slice_read")
                    operandWords = std::min(
                        operandWords,
                        mlir::cast<mlir::MemRefType>(inst.getOperand(0).getType())
                                .getElementTypeBitWidth() /
                            32.);
                carries = std::max<unsigned>(
                    carries,
                    llvm::count_if(loop.getResultTypes(), [](mlir::Type t) {
                      return mlir::isa<mlir::ShapedType>(t);
                    }));
                if (!caps->operandGroupRows)
                  continue;
                llvm::SmallPtrSet<mlir::Operation *, 4> scores;
                for (auto &s : loop.getBody()->without_terminator()) {
                  if (s.getName().stripDialect() != "lane_sum")
                    continue;
                  auto *consumer =
                      s.hasOneUse() ? *s.getUsers().begin() : nullptr;
                  // Dot products summed directly produce one live score.
                  scores.insert(consumer &&
                                        mlir::isa<mlir::arith::AddFOp>(consumer)
                                    ? consumer
                                    : &s);
                }
                projections = std::max<unsigned>(projections, scores.size());
              }
              auto streams = caps->operandGroupRows ? projections : carries;
              if (sweepBytesPerOwner) {
                // An operand issue group streams both score products through
                // the same staging area. Independent accumulator streams
                // otherwise retain their own pair of fragments.
                double operandStreams =
                    caps->operandGroupRows ? 1. : std::max(1u, carries);
                double slots =
                    operandWords * operandStreams *
                        (physicalRows + span.getInt()) * lanes +
                    std::max(2u, carries) * physicalRows * span.getInt() +
                    carries * physicalRows * lanes;
                auto file =
                    ctx.target.services.facts.find("register_file")
                        ? intentdsl::quantity(ctx, "register_file", "reg32")
                        : std::nullopt;
                double budget = caps->residentRegisters
                                    ? caps->residentRegisters
                                    : file.value_or(0) / 2;
                // Resident matrix A/B operands, two score intermediates,
                // and each live FP32 accumulator occupy the CTA register file.
                // Excess live words need a store and reload per issue.
                double spill = budget ? std::max(0., slots - budget) : 0.;
                spillBytes =
                    panelBytes * 8. * spill / (span.getInt() * lanes * 4.);
                e.terms["live_fragment_words"] = {slots, "reg32"};
                e.terms["excess_fragment_bytes"] = {spillBytes, "B"};
              }
              if (streams > 1 && caps->exchangeWidth &&
                  caps->residentRegisters) {
                auto rowGroup = caps->operandGroupRows ? caps->operandGroupRows
                                                       : caps->exchangeWidth;
                double packets =
                    std::ceil(double(span.getInt()) / caps->exchangeWidth);
                double slots =
                    operandWords * streams * (physicalRows + span.getInt()) *
                        lanes +
                    streams * physicalRows * span.getInt() +
                    carries * physicalRows * lanes;
                issueGroups = std::ceil(physicalRows / rowGroup) *
                              std::ceil(double(lanes) / 32.);
                dependency *= (1. + (packets - 1.) * streams) *
                              std::max(1., slots / caps->residentRegisters);
                e.terms["concurrent_accumulator_slots"] = {slots, "reg32"};
                e.terms["score_exchange_packets"] = {packets, "count"};
              }
              e.terms["matrix_issue_groups"] = {issueGroups, "count"};
              e.terms["shared_panel_read_bytes"] = {panelBytes, "B"};
            }
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
            auto shared = isSharedMode || isMatrixFold;
            for (auto *load : r->slices) {
              auto loop = load->getParentOfType<mlir::scf::ForOp>();
              if (loop && loop->hasAttr("kir.reassociate") && sweepBytesPerOwner)
                continue;
              auto v = load->getResult(0);
              auto axis = intentdsl::kir::memberAxis(v);
              const auto *shape = f->shapeOf(axis->first);
              auto bytes = (*shape)[axis->second].getAsInteger().value() *
                           mlir::cast<mlir::ShapedType>(v.getType())
                               .getElementTypeBitWidth() /
                           8.;
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
          double waveSetup = 0;
          if (!span && choice.getAs<mlir::StringAttr>("memory") == "direct") {
            dependency *= 1. + std::max(0., double(w) * lanes / 4096. - 1.);
            waveSetup = waves;
          }
          if (launch && sync)
            e.timeNs = *launch + *sync * (waves * issueGroups * dependency +
                                          waveSetup +
                                          blocks / std::max<double>(1, n));
          if (e.timeNs && readCost)
            if (auto rate = intentdsl::quantity(ctx, "read_1", "B/ns"))
              *e.timeNs += readCost / *rate;
          if (e.timeNs && panelBytes)
            if (auto rate = intentdsl::quantity(ctx, "read_1", "B/ns"))
              *e.timeNs += panelBytes / intentdsl::quantity(
                                            ctx, "read_1_shared_bw", "B/ns")
                                            .value_or(*rate) +
                           spillBytes / *rate;
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
