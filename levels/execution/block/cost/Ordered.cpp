#include "intentdsl/levels/execution/block/ir/FoldIssue.h"
#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
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
          int64_t triLanes = 1;
          for (auto loop : op->getRegion(0).front().getOps<mlir::scf::ForOp>()) {
            intentdsl::block::TriangularRecurrenceInfo tri;
            if (intentdsl::block::matchTriangularRecurrence(
                    loop, op->getParentOfType<mlir::func::FuncOp>(), f, &tri))
              triLanes = std::max<int64_t>(
                  triLanes,
                  llvm::PowerOf2Ceil(std::max<int64_t>(2, tri.extent)));
          }
          int64_t foldLanes = intentdsl::block::foldBodyLanes(op, *f);
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
          auto span = choice.getAs<mlir::IntegerAttr>("span");
          bool isMatrixFoldEarly =
              choice.getAs<mlir::StringAttr>("memory") == "matrix-fold";
          double issueGroups = std::ceil(
              double(w) *
              (span && !isMatrixFoldEarly ? foldLanes
                                          : std::max(lanes, triLanes)) /
              issue);
          double dependency = 1;
          // A slice indexed by the fold induction variable is loaded once per
          // visited token per tile of owners. Use its actual access dimension,
          // not the number of tokens in a single issue packet.
          double sweepBytesPerOwner = 0, maxTokenBytes = 0;
          bool relaxedNormCost =
              intentdsl::block::hasReassociativeNormalization(*op);
          llvm::SmallVector<mlir::scf::ForOp, 4> reassociateLoops;
          for (auto loop :
               op->getRegion(0).front().getOps<mlir::scf::ForOp>())
            if (loop->hasAttr("kir.reassociate"))
              reassociateLoops.push_back(loop);
          bool fuseTwoPassNorm =
              relaxedNormCost && reassociateLoops.size() == 2 &&
              reassociateLoops[0].getNumResults() == 1 &&
              !mlir::isa<mlir::ShapedType>(
                  reassociateLoops[0].getResultTypes()[0]) &&
              llvm::any_of(reassociateLoops[1].getResultTypes(),
                           [](mlir::Type t) {
                             return mlir::isa<mlir::ShapedType>(t);
                           });
          for (auto [loopIdx, loop] : llvm::enumerate(reassociateLoops)) {
            if (fuseTwoPassNorm && loopIdx == 0)
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
              if (slice) {
                if (int64_t page =
                        intentdsl::block::alignedPageExtent(inst, loop, *f, 1)) {
                  int64_t depth =
                      (*sh)[inst.getAttrOfType<mlir::IntegerAttr>("axis")
                                .getInt()]
                          .getAsInteger()
                          .value_or(1);
                  tokenBytes +=
                      depth *
                      mlir::cast<mlir::MemRefType>(inst.getOperand(0).getType())
                          .getElementTypeBitWidth() /
                      8.;
                  int64_t batches =
                      (*f->shapeOf(op->getOperand(0)))[0]
                          .getAsInteger()
                          .value_or(1);
                  extent = std::max(
                      extent,
                      std::max<int64_t>(
                          1, (*sh)[0].getAsInteger().value_or(0) * page /
                                 std::max<int64_t>(1, batches)));
                  continue;
                }
              }
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
            maxTokenBytes = std::max(maxTokenBytes, tokenBytes);
            // For a causal one-to-one owner axis the mean prefix or suffix
            // length is (L+1)/2. Otherwise count the full sweep.
            double visits = extent;
            bool relaxedNorm =
                intentdsl::block::hasReassociativeNormalization(*op);
            auto iv = mlir::dyn_cast<mlir::BlockArgument>(loop.getLowerBound());
            if (!iv)
              iv = mlir::dyn_cast_if_present<mlir::BlockArgument>(
                  intentdsl::block::causalPrefixOwner(loop));
            if (!iv && relaxedNorm)
              iv = mlir::dyn_cast_if_present<mlir::BlockArgument>(
                  intentdsl::block::causalMaskOwner(loop));
            if (iv && iv.getOwner() == loop->getBlock()) {
              auto axes = op->getAttrOfType<mlir::DenseI64ArrayAttr>("axes");
              auto sh = f->shapeOf(op->getOperand(iv.getArgNumber()));
              if (axes && sh &&
                  (*sh)[axes[iv.getArgNumber()]].getAsInteger() == extent &&
                  (!span || span.getInt() < extent))
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
            auto lanesPerPanel = span.getInt() * foldLanes;
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
              if (caps && caps->panelBytes > 0 &&
                  blocks >= 2.0 * std::max<int64_t>(1, f->participants) &&
                  2.0 * span.getInt() * maxTokenBytes > caps->panelBytes / 2.0)
                panelBytes *= 1.25;
              dependency =
                  (1. + std::max(0., live / 32768. - 1.)) / span.getInt();
              // Independent matrix updates compete for resident accumulator
              // fragments and exchange score columns between issue groups.
              unsigned carries = 0, projections = 0;
              double operandWords = 1.;
              bool indirectSlice = false;
              for (auto loop :
                   op->getRegion(0).front().getOps<mlir::scf::ForOp>()) {
                if (!loop->hasAttr("kir.reassociate"))
                  continue;
                for (auto &inst : loop.getBody()->without_terminator())
                  if (inst.getName().stripDialect() == "slice_read") {
                    operandWords = std::min(
                        operandWords,
                        mlir::cast<mlir::MemRefType>(inst.getOperand(0).getType())
                                .getElementTypeBitWidth() /
                            32.);
                    if (!intentdsl::block::rectangularTokenLoad(inst, loop) &&
                        !intentdsl::block::alignedPageExtent(inst, loop, *f,
                                                             span.getInt()))
                      indirectSlice = true;
                  }
                carries = std::max<unsigned>(
                    carries,
                    llvm::count_if(loop.getResultTypes(), [](mlir::Type t) {
                      return mlir::isa<mlir::ShapedType>(t);
                    }));
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
              auto streams = caps->operandGroupRows
                                 ? projections
                                 : std::max(carries, projections);
              if (sweepBytesPerOwner) {
                // An operand issue group streams both score products through
                // the same staging area. Independent register-fed streams
                // otherwise retain their own pair of operand fragments and
                // narrowed weight fragment.
                double operandStreams =
                    caps->operandGroupRows ? 1. : std::max(1u, streams);
                double scoreSlots = (!caps->operandGroupRows && projections > 1)
                                        ? double(projections + 1u)
                                        : double(std::max(2u, carries));
                double slots =
                    operandWords * operandStreams *
                        (physicalRows + span.getInt()) * lanes +
                    scoreSlots * physicalRows * span.getInt() +
                    carries * physicalRows * lanes +
                    (indirectSlice ? double(span.getInt()) * lanes : 0.);
                auto file =
                    ctx.target.services.facts.find("register_file")
                        ? intentdsl::quantity(ctx, "register_file", "reg32")
                        : std::nullopt;
                double budget = caps->residentRegisters
                                    ? caps->residentRegisters
                                    : file.value_or(0) / 2;
                if (budget && physicalRows >= 128 &&
                    caps->maxWarpgroups >= 2 && carries <= 1 &&
                    projections <= 1)
                  budget *= 1.75;
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
            double independentLoads = 0, wordsPerStep = 0, carryWords = 0;
            for (auto loop :
                 op->getRegion(0).front().getOps<mlir::scf::ForOp>()) {
              for (auto t : loop.getResultTypes()) {
                if (auto st = mlir::dyn_cast<mlir::ShapedType>(t))
                  carryWords += lanes * (st.getElementTypeBitWidth() / 32.0);
                else
                  carryWords += t.getIntOrFloatBitWidth() / 32.0;
              }
              loop.walk([&](mlir::Operation *v) {
                if (v->getName().stripDialect() == "slice_read") {
                  ++independentLoads;
                  double elemWords =
                      mlir::cast<mlir::ShapedType>(v->getResult(0).getType())
                          .getElementTypeBitWidth() /
                      32.0;
                  wordsPerStep += lanes * elemWords;
                } else if (v->getName().getStringRef() == "memref.load") {
                  ++independentLoads;
                  double elemWords =
                      v->getResult(0).getType().getIntOrFloatBitWidth() / 32.0;
                  wordsPerStep += elemWords;
                }
              });
            }
            double live = w * (carryWords + span * wordsPerStep);
            dependency = (1 + independentLoads / span) *
                         (1 + std::max(0., live - 6144) / 2048);
            e.terms["grouped_read_elements"] = {live, "count"};
          }
          double readCost = 0;
          if (auto r = intentdsl::block::sharedReads(op, *f)) {
            bool innerUniform =
                w > 1 &&
                r->axis + 1 ==
                    op->getRegion(0).front().getNumArguments() &&
                r->group % w == 0;
            auto shared = isSharedMode || isMatrixFold || innerUniform;
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
          if (choice.getAs<mlir::StringAttr>("memory") == "scatter-fold") {
            int64_t split = choice.getAs<mlir::IntegerAttr>("split").getInt();
            intentdsl::block::AssociativeScatterLoopInfo assocInfo;
            intentdsl::block::canSplitAssociativeScatter(op, f, w, split,
                                                         &assocInfo);
            double totalBlocks = blocks * double(split);
            double residentPerSm =
                std::clamp(1024.0 / std::max<int64_t>(32, w), 2.0, 8.0);
            double concurrentBlocks = std::max<double>(
                1.0, std::max<int64_t>(1, f->participants) * residentPerSm);
            waves = std::ceil(totalBlocks / concurrentBlocks);
            e.terms["independent_blocks"] = {totalBlocks, "count"};
            e.terms["issue_waves"] = {waves, "count"};
            double itemsPerCta =
                std::ceil(double(std::max<int64_t>(1, assocInfo.extent)) /
                          double(split));
            double panelN = (w >= 16 && assocInfo.extent >= 32) ? 8.0 : 1.0;
            double stepsPerCta = std::ceil(itemsPerCta / panelN);
            double activeSlots = std::min<double>(totalBlocks, concurrentBlocks);
            double smUnderutil = concurrentBlocks / activeSlots;
            double sectorPenalty =
                32.0 / std::min<double>(32.0, double(w) * 4.0);
            dependency = stepsPerCta * smUnderutil * sectorPenalty;
            waveSetup = 0.8 * waves;
            double atomicBytes =
                double(assocInfo.extent) *
                (double(n) * 4.0 * sectorPenalty * 2.0 + 4.0 * blocks);
            readCost += atomicBytes;
            panelBytes = 0;
          }
          if (!span && choice.getAs<mlir::StringAttr>("memory") == "direct") {
            auto fnOp = op->getParentOfType<mlir::func::FuncOp>();
            dependency *= 1. + std::max(0., double(w) * std::max(lanes, triLanes) /
                                                4096. -
                                                1.);
            double panelBucketSteps = 0;
            bool hasPanelBucket = false;
            for (auto loop :
                 op->getRegion(0).front().getOps<mlir::scf::ForOp>()) {
              intentdsl::block::OrderedSelectionInfo selInfo;
              if (intentdsl::block::matchOrderedSelectionChain(loop,
                                                               &selInfo)) {
                int64_t ext =
                    intentdsl::block::orderedStaticLoopExtent(loop, fnOp, f)
                        .value_or(1024);
                int64_t panelW = std::clamp<int64_t>(
                    llvm::PowerOf2Ceil(std::max<int64_t>(1, ext)), 64, 1024);
                if (w == 1)
                  dependency *= std::max<double>(1.0, 2.0 * selInfo.k) / panelW;
                else
                  dependency *= std::max<double>(1.0, double(ext) / 16.0);
                continue;
              }
              intentdsl::block::TriangularRecurrenceInfo triInfo;
              if (intentdsl::block::matchTriangularRecurrence(loop, fnOp, f,
                                                              &triInfo)) {
                double regBudget = (caps && caps->residentRegisters)
                                       ? double(caps->residentRegisters)
                                       : 32768.0;
                double residentPerSm = std::clamp(
                    regBudget /
                        std::max(1.0, double(w) * double(triLanes) * 2.0),
                    1.0, 4.0);
                double sms = std::max<double>(1.0, f->participants);
                double concurrentBlocks =
                    std::max<double>(1.0, sms * residentPerSm);
                double activeCtas = std::min<double>(blocks, concurrentBlocks);
                double latencyHide = std::min<double>(
                    2.0, std::max<double>(1.0, activeCtas / sms));
                bool hasNativeGather = caps && caps->nativeTileGather;
                // Without native tile gather, each step reduces a full [w, K]
                // tile per CTA; fewer CTAs halve both reduction barriers and
                // shared column loads. With native gather, extra resident CTAs
                // per SM hide serial loop latency directly.
                double reduceOverhead = hasNativeGather
                                            ? (1.0 / latencyHide)
                                            : (1.0 + 4.0 / double(std::max<int64_t>(1, w))) /
                                                  std::sqrt(latencyHide);
                waves = std::ceil(blocks / sms);
                e.terms["issue_waves"] = {waves, "count"};
                dependency *=
                    std::max<double>(1.0, double(triInfo.extent) / 64.0) *
                    reduceOverhead;
                double sharedBatches = std::min<double>(
                    blocks,
                    hasNativeGather
                        ? std::max<double>(1.0, double(n) / 64.0)
                        : blocks);
                double coeffBytes = 0.5 * double(triInfo.extent) *
                                    double(triInfo.extent + 1) * 4.0 *
                                    sharedBatches;
                double sectorPenalty =
                    32.0 / std::min<double>(32.0, double(w) * 4.0);
                double stateBytes =
                    2.0 * double(triInfo.extent) * 4.0 * double(n) *
                    sectorPenalty;
                readCost += coeffBytes + stateBytes;
                continue;
              }
              intentdsl::block::ExclusiveScanLoopInfo scanInfo;
              if (intentdsl::block::matchExclusiveScanLoop(loop, fnOp, f,
                                                           &scanInfo)) {
                hasPanelBucket = true;
                panelBucketSteps += (w == 1) ? 1.0 : double(scanInfo.extent);
                continue;
              }
              intentdsl::block::BoundedHistogramLoopInfo histInfo;
              if (intentdsl::block::matchBoundedHistogramLoop(loop, fnOp, f,
                                                              &histInfo)) {
                hasPanelBucket = true;
                int64_t ext =
                    intentdsl::block::orderedStaticLoopExtent(loop, fnOp, f)
                        .value_or(1024);
                panelBucketSteps +=
                    (w == 1) ? std::ceil(double(ext) / 64.0) : double(ext);
                continue;
              }
              intentdsl::block::BoundedScatterLoopInfo scatInfo;
              if (intentdsl::block::matchBoundedScatterLoop(loop, fnOp, f,
                                                            &scatInfo)) {
                hasPanelBucket = true;
                int64_t ext =
                    intentdsl::block::orderedStaticLoopExtent(loop, fnOp, f)
                        .value_or(1024);
                panelBucketSteps +=
                    (w == 1) ? std::ceil(double(ext) / 32.0) : double(ext);
                continue;
              }
              if (lanes == 1 && triLanes == 1 &&
                  loop.getNumRegionIterArgs() == 0) {
                bool hasIndirectStore = false;
                for (auto &inst : loop.getBody()->without_terminator())
                  if (auto st = mlir::dyn_cast<mlir::memref::StoreOp>(inst))
                    if (st.getIndices().empty() ||
                        st.getIndices().back() != loop.getInductionVar())
                      hasIndirectStore = true;
                if (hasIndirectStore) {
                  if (auto ext = intentdsl::block::orderedStaticLoopExtent(
                          loop, fnOp, f);
                      ext && *ext >= 64) {
                    double warpsInCta = std::ceil(double(w) / 32.0);
                    double activeSm = std::min<double>(
                        blocks, std::max<int64_t>(1, f->participants));
                    double smUnderutil =
                        std::max<double>(1, f->participants) / activeSm;
                    double sectorPenalty =
                        16.0 / std::min<double>(16.0, double(w) * 4.0);
                    dependency *= warpsInCta * smUnderutil * sectorPenalty *
                                  (loop->hasAttr("kir.reassociate")
                                       ? double(*ext)
                                       : double(*ext) / 64.0);
                  }
                }
              }
            }
            if (hasPanelBucket)
              dependency *= std::max<double>(1.0, panelBucketSteps);
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
