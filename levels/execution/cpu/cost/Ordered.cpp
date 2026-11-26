#include "intentdsl/levels/execution/cpu/ir/OrderedValue.h"
#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
void ownerCost(Unit &u) {
  u.costModels.push_back(
      {{"ordered-tasks", u.qualify("ordered-owners"), "2",
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
          if (panel) {
            bool retained =
                cache && (cache.get("scores") ||
                          (cache.get("window") &&
                           cache.getAs<mlir::IntegerAttr>("window").getInt() >=
                               panel->keys));
            double dots = double(n) * panel->keys * (retained ? 1 : 2);
            e.terms["score_dot_products"] = {dots, "count"};
            double flop = 2. * dots * panel->depth;
            if (shared) {
              auto dot = intentdsl::cpu::orderedDotPanel(*o, *f, shared->axis,
                                                         rm->packedOwners);
              auto value = dot ? intentdsl::cpu::orderedValuePanel(*o, *f, *dot)
                               : std::nullopt;
              if (value) {
                double valueFlop = 2. * n * value->tokens * value->columns;
                e.terms["value_contraction_work"] = {valueFlop, "FLOP"};
                bool matrixPV = cache && cache.contains("value_width");
                double width =
                    matrixPV ? cache.getAs<mlir::IntegerAttr>("batch").getInt()
                             : 1;
                *e.timeNs += valueFlop / (width * rateVal);
              }
            }
            if (cache && cache.get("engine")) {
              auto batch = cache.getAs<mlir::IntegerAttr>("batch").getInt();
              int64_t tokens =
                  cache.getAs<mlir::IntegerAttr>("score_tile").getInt();
              double panels = std::ceil(double(n) / batch) *
                              std::ceil(double(panel->keys) / tokens);
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
                e.terms["strided_pack_elements"] = {packedElements, "count"};
                *e.timeNs += 3 * packedElements / scalarRate;
              }
              double contexts = tasks;
              e.terms["brgemm_contexts"] = {contexts, "count"};
              double transition = 4 * launchVal + 2 * sync;
              *e.timeNs +=
                  flop / (batch * rateVal) + 2 * pack / bwVal +
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
            double trip = panel ? double(panel->keys) : 1.;
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
                      8. * trip * n;
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
