#include "intentdsl/levels/execution/cpu/cost/PanelLayout.h"
#include "intentdsl/levels/execution/cpu/ir/OrderedValue.h"
#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
namespace {
// A masked maximum over token <= owner visits the prefix of the last owner
// in the batch. Each downstream matrix panel rounds that prefix to its tile.
double meanKeySpan(const intentdsl::cpu::ScoreCachePlan &panel,
                   mlir::Value owner, int64_t owners, int64_t batch,
                   int64_t tile) {
  using namespace mlir;
  auto loop = panel.record->getParentOfType<scf::ForOp>();
  if (!loop || loop.getNumRegionIterArgs() != 1 || owners <= 0 ||
      owners % batch || !llvm::isPowerOf2_64(tile) ||
      !llvm::isPowerOf2_64(batch))
    return panel.keys;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto maximum = yield.getOperand(0).getDefiningOp<arith::MaxNumFOp>();
  if (!maximum)
    return panel.keys;
  Value carry = loop.getRegionIterArgs().front();
  Value masked = maximum.getLhs() == carry   ? maximum.getRhs()
                 : maximum.getRhs() == carry ? maximum.getLhs()
                                             : Value{};
  if (!masked)
    return panel.keys;
  Value pred, inactive;
  if (auto branch = masked.getDefiningOp<scf::IfOp>();
      branch && !branch.getElseRegion().empty()) {
    pred = branch.getCondition();
    inactive = branch.elseYield().getOperand(0);
  } else if (auto sel = masked.getDefiningOp<arith::SelectOp>()) {
    pred = sel.getCondition();
    inactive = sel.getFalseValue();
  }
  auto cmp = pred ? pred.getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
  auto cst = inactive ? inactive.getDefiningOp<arith::ConstantOp>()
                      : arith::ConstantOp{};
  auto value = cst ? dyn_cast<FloatAttr>(cst.getValue()) : FloatAttr{};
  if (!cmp || !value || !value.getValue().isNegInfinity())
    return panel.keys;
  Value iv = loop.getInductionVar();
  bool prefix = (cmp.getLhs() == owner && cmp.getRhs() == iv &&
                 cmp.getPredicate() == arith::CmpIPredicate::sge) ||
                (cmp.getLhs() == iv && cmp.getRhs() == owner &&
                 cmp.getPredicate() == arith::CmpIPredicate::sle);
  if (!prefix)
    return panel.keys;
  // Sum ceil(min(j * batch, K) / tile) over complete query batches j.
  // Group equal rounded extents instead of scanning a potentially long axis.
  int64_t groups = owners / batch;
  int64_t growing = std::min(groups, (panel.keys - 1) / batch);
  double total;
  if (batch >= tile)
    total = double(batch) * growing * (growing + 1) * .5;
  else {
    int64_t equal = tile / batch, runs = growing / equal,
            rest = growing % equal;
    total = double(tile) * (double(equal) * runs * (runs + 1) * .5 +
                            double(rest) * (runs + 1));
  }
  total += double(groups - growing) * llvm::divideCeil(panel.keys, tile) * tile;
  return total / groups;
}
} // namespace
void ownerCost(Unit &u) {
  u.costModels.push_back(
      {{"ordered-tasks", u.qualify("ordered-owners"), "5",
        [](mlir::Operation *o, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto f = ctx.target.services.get<intentdsl::CallFacts>();
          auto n = intentdsl::kir::ownerCount(o, *f);
          auto cache = mlir::dyn_cast<mlir::DictionaryAttr>(c.value);
          auto w = mlir::cast<mlir::IntegerAttr>(cache ? cache.get("grain")
                                                       : c.value)
                       .getInt();
          auto rm = ctx.target.services.get<intentdsl::cpu::RowMachine>();
          auto workers = rm->workers;
          double tasks = llvm::divideCeil(n, w),
                 waves = std::ceil(tasks / workers);
          CostEstimate e;
          e.scope = "full_call";
          e.bottleneck = "owner task granularity";
          e.terms["native_tasks"] = {tasks, "count"};
          e.terms["worker_waves"] = {waves, "count"};
          double launchVal =
              intentdsl::quantity(ctx, "launch", "ns").value_or(4000.);
          double rateVal = intentdsl::quantity(ctx, "vector", "FLOP/ns")
                               .value_or(std::max(1., (rm->vectorBits / 32.) *
                                                          2. * workers));
          double bwVal = intentdsl::quantity(ctx, "read_1", "B/ns")
                             .value_or(std::max(8., 8. * workers));
          e.timeNs = launchVal +
                     waves * (1 + std::min(w, n)) * workers / rateVal +
                     tasks / rateVal;
          auto panel = intentdsl::cpu::reusableScores(
              o, ctx.target, intentdsl::cpu::scoreSlotLimit(*rm));
          auto shared = intentdsl::cpu::sharedInterval(o, *f);
          double keySpan = panel ? panel->keys : 1.;
          int64_t batch = cache && cache.contains("batch")
                              ? cache.getAs<mlir::IntegerAttr>("batch").getInt()
                              : 1;
          int64_t tile =
              cache && cache.contains("score_tile")
                  ? cache.getAs<mlir::IntegerAttr>("score_tile").getInt()
                  : rm->orderedDotTokens;
          // The streamed backend uses a sequence of hardware tiles; extra
          // owners beyond AMX's 16 physical tile rows issue more multiplies.
          constexpr int64_t amxTileRows = 16;
          double matrixWidth =
              rm->orderedDotTokens ? std::min(batch, amxTileRows) : batch;
          if (panel && shared && rm->orderedDotTokens && cache &&
              cache.contains("window")) {
            auto owner = o->getRegion(0).front().getArgument(shared->axis);
            keySpan = meanKeySpan(*panel, owner, shared->extent, batch, tile);
            e.terms["mean_key_span"] = {keySpan, "count"};
          }
          if (panel) {
            bool retained =
                cache && (cache.get("scores") ||
                          (cache.get("window") &&
                           cache.getAs<mlir::IntegerAttr>("window").getInt() >=
                               panel->keys));
            double dots = double(n) * keySpan * (retained ? 1 : 2);
            e.terms["score_dot_products"] = {dots, "count"};
            double flop = 2. * dots * panel->depth;
            auto dot = shared ? intentdsl::cpu::orderedDotPanel(
                                    *o, *f, shared->axis, rm->packedOwners)
                              : std::nullopt;
            auto value = dot ? intentdsl::cpu::orderedValuePanel(*o, *f, *dot)
                             : std::nullopt;
            if (shared) {
              if (value) {
                double valueFlop = 2. * n * keySpan * value->columns;
                e.terms["value_contraction_work"] = {valueFlop, "FLOP"};
                bool matrixPV = cache && cache.contains("value_width");
                *e.timeNs +=
                    valueFlop / ((matrixPV ? matrixWidth : 1.) * rateVal);
              }
            }
            if (cache && cache.get("engine")) {

              int64_t tokens =
                  cache.getAs<mlir::IntegerAttr>("score_tile").getInt();
              double tileTrips = keySpan == panel->keys
                                     ? std::ceil(keySpan / tokens)
                                     : keySpan / tokens;
              double panels = std::ceil(double(n) / batch) * tileTrips;
              double pack = panels * batch * panel->depth * sizeof(uint16_t);
              // AMX reduces BF16 pairs across a complete owner tile. Every
              // invocation pays the hardware context transition and packing.
              double sync =
                  intentdsl::quantity(ctx, "sync", "ns").value_or(launchVal);
              e.terms["brgemm_calls"] = {panels, "count"};
              e.terms["query_pack_bytes"] = {pack, "B"};
              // A dense downstream tile is admitted and transposed to VNNI
              // with scalar addressing and a padded destination. The native
              // owner task instead keeps its transposed query panel across
              // keys.
              if (rm->orderedDotTokens) {
                double packedElements = panels * tokens * panel->depth;
                double scalarRate = rateVal / (rm->vectorBits / 32);
                auto packingWork = [&](mlir::Operation *read, bool transpose,
                                       double repeated) {
                  if (!value || !cache.contains("value_width"))
                    return repeated;
                  auto arg =
                      mlir::cast<mlir::BlockArgument>(read->getOperand(0));
                  int64_t once = intentdsl::cpu::invocationPanelElements(
                      *o, *f, *rm, arg, tokens, batch, transpose);
                  if (!once)
                    return repeated;
                  e.terms[transpose ? "shared_key_pack_elements"
                                    : "shared_value_pack_elements"] = {
                      double(once), "count"};
                  *e.timeNs +=
                      (2. * once + repeated) * sizeof(uint16_t) / bwVal + sync;
                  return double(once);
                };
                if (dot)
                  packedElements = packingWork(dot->key, true, packedElements);
                e.terms["strided_pack_elements"] = {packedElements, "count"};
                *e.timeNs += 3 * packedElements / scalarRate;
                if (auto cols = cache.getAs<mlir::IntegerAttr>("value_width")) {
                  // Two separately lowered vendor calls exchange FP32 scores
                  // and BF16 weights through dense payload and validity arrays.
                  double scoreEntries = panels * batch * tokens;
                  double valueEntries = panels * tokens * cols.getInt();
                  double packedBytes =
                      2. * scoreEntries *
                      (sizeof(float) + sizeof(uint16_t) + 2 * sizeof(uint8_t));
                  double gather =
                      intentdsl::quantity(ctx, "copy_gather_1_bw", "B/ns")
                          .value_or(bwVal);
                  double scatter =
                      intentdsl::quantity(ctx, "copy_scatter_1_bw", "B/ns")
                          .value_or(bwVal);
                  double outBytes = double(batch) * cols.getInt() *
                                    (sizeof(float) + sizeof(uint8_t));
                  // The score and value projections execute in different loops.
                  double scoreLive =
                      double(batch) * tokens *
                          (sizeof(float) + sizeof(uint8_t)) +
                      double(batch + tokens) * panel->depth * sizeof(uint16_t);
                  double valueLive =
                      outBytes +
                      double(batch) * tokens *
                          (sizeof(uint16_t) + sizeof(uint8_t)) +
                      double(tokens) * cols.getInt() * sizeof(uint16_t);
                  bool online = cache.contains("online");
                  if (online) {
                    // The projected QK tile feeds the weight expression in
                    // the same iteration. Its pointwise rescaling of C walks
                    // contiguous private storage before BRGEMM reads C.
                    packedBytes -= 2 * scoreEntries * sizeof(float);
                    auto read =
                        intentdsl::quantity(ctx, "read_1_private_bw", "B/ns")
                            .value_or(bwVal);
                    auto write =
                        intentdsl::quantity(ctx, "write_1_private_bw", "B/ns")
                            .value_or(bwVal);
                    *e.timeNs += panels * outBytes * (1. / read + 1. / write);
                    e.terms["rescale_bytes"] = {2 * panels * outBytes, "B"};
                    double active =
                        batch * tokens * (sizeof(float) + sizeof(uint16_t) + 2);
                    double coldBytes = std::max(0., active - rm->l1CacheBytes);
                    packedBytes += 2 * panels * coldBytes;
                    // Single-use pointwise chains share one packet loop. The
                    // FP32 weights feeding both mass and PV still need their
                    // private payload and validity read/write.
                    if (batch * tokens > rm->maxVectorElements)
                      packedBytes +=
                          2 * scoreEntries * (sizeof(float) + sizeof(uint8_t));
                    if (batch * cols.getInt() > rm->maxVectorElements)
                      packedBytes += 2 * panels * outBytes;
                  }
                  double liveBytes = std::max(scoreLive, valueLive);
                  double spill = std::max(0., liveBytes - rm->ownerStackBytes);
                  e.terms["matrix_spill_bytes"] = {2 * panels * spill, "B"};
                  if (value)
                    valueEntries =
                        packingWork(value->read, false, valueEntries);
                  e.terms["value_vnni_pack_elements"] = {valueEntries, "count"};
                  e.terms["matrix_exchange_bytes"] = {packedBytes, "B"};
                  e.terms["matrix_tile_live_bytes"] = {liveBytes, "B"};
                  e.terms["brgemm_calls"] = {2 * panels, "count"};
                  *e.timeNs += 3 * valueEntries / scalarRate +
                               .5 * packedBytes * (1. / gather + 1. / scatter) +
                               2 * panels * spill / bwVal +
                               panels * sync * .15 / workers;
                }
              }
              double contexts = tasks;
              e.terms["brgemm_contexts"] = {contexts, "count"};
              double transition = 4 * launchVal + 2 * sync;
              *e.timeNs +=
                  flop / (matrixWidth * rateVal) + 2 * pack / bwVal +
                  (transition * contexts + panels * sync * 0.15) / workers;
            } else
              *e.timeNs += flop / rateVal;
            if (!shared)
              *e.timeNs += ((retained ? 0. : panel->readBytes) +
                            (retained ? 8. : 0.)) *
                           n * panel->keys / bwVal;
          }
          if (shared) {
            bool together = cache && cache.get("inner");
            auto ownersPerLoad =
                together ? cache.getAs<mlir::IntegerAttr>("batch").getInt()
                         : int64_t(1);
            double trip = keySpan;
            int64_t regCarryFloats = 0;
            auto evalBound = [&](mlir::Value v) -> std::optional<int64_t> {
              if (auto cOp = v.getDefiningOp<mlir::arith::ConstantIndexOp>())
                return cOp.value();
              if (auto dOp = v.getDefiningOp<mlir::memref::DimOp>()) {
                if (auto ax = dOp.getConstantIndex()) {
                  if (auto *sh = f->shapeOf(dOp.getSource());
                      sh && *ax < sh->size())
                    return (*sh)[*ax].getAsInteger();
                }
              }
              return std::nullopt;
            };
            for (auto loop : o->getRegion(0).front().getOps<mlir::scf::ForOp>()) {
              if (!panel) {
                auto lb = evalBound(loop.getLowerBound());
                auto ub = evalBound(loop.getUpperBound());
                if (lb && ub && *ub > *lb)
                  trip = std::max(trip, double(*ub - *lb));
              }
              for (mlir::Value init : loop.getInitArgs()) {
                if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(init.getType())) {
                  int64_t k = rt.getNumElements();
                  if (k > 0 && k < 64)
                    regCarryFloats += k;
                }
              }
            }
            auto bytes = shared->readBytes * trip * n / ownersPerLoad;
            // One of the two projections is replaced by a private FP32
            // store/load pair for the score of each owner.
            if (together && cache.get("window")) {
              double savedPerStep =
                  panel ? std::min(shared->readBytes, panel->readBytes)
                        : shared->readBytes * .5;
              bytes = (shared->readBytes - savedPerStep) * trip * n /
                          ownersPerLoad +
                      (cache.contains("online") ? 0. : 8. * trip * n);
            }
            int64_t regCapFloats = (rm->vectorBits >= 512) ? 256 : 128;
            if (together && regCarryFloats * ownersPerLoad > regCapFloats) {
              double spillBytesPerStep =
                  double(regCarryFloats * ownersPerLoad - regCapFloats) * 8.;
              bytes += spillBytesPerStep * trip * (double(n) / ownersPerLoad);
            }
            double activeWorkers =
                std::max(1.0, std::min<double>(workers, tasks));
            double effBw =
                bwVal * (activeWorkers / std::max<double>(1, workers));
            e.terms["common_input_bytes_per_member"] = {bytes, "B"};
            *e.timeNs += bytes / effBw;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
