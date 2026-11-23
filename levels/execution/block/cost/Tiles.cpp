#include "intentdsl/levels/algorithm/kir/ir/Permutation.h"
#include "intentdsl/levels/algorithm/kir/ir/Traffic.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "llvm/ADT/DenseSet.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerCost(Unit &u) {
  u.costModels.push_back(
      {{"tile-time", u.qualify("structure"), "1",
        [](Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto *f = ctx.target.services.get<intentdsl::CallFacts>();
          auto [m, n] = f->geometry(op);
          auto choice = cast<DictionaryAttr>(c.value);
          bool stream = choice.getAs<StringAttr>("topology") == "stream";
          CostEstimate e;
          e.scope = "block program";
          e.bottleneck = "global memory";
          auto read = intentdsl::quantity(ctx, "read_1", "B/ns"),
               write = intentdsl::quantity(ctx, "write_1", "B/ns"),
               strided = intentdsl::quantity(ctx, "write_16", "B/ns"),
               launch = intentdsl::quantity(ctx, "launch", "ns");
          bool swapped = (choice.get("walk") &&
                          choice.getAs<StringAttr>("walk") == "destination");
          if (swapped)
            std::swap(m, n);
          auto stridedRead = intentdsl::quantity(ctx, "read_16", "B/ns");
          double r = 0, w = 0, ws = 0;
          unsigned reductions = 0, summaryFields=0, scans=0, atomics=0;
          op->walk([&](Operation *i) {
            auto k = i->getName().stripDialect();
            reductions += k == "sum" || k == "max_num" || k=="welford";
            summaryFields += k=="welford" ? 3 : 0;
            scans += k=="scan_sum";
            atomics += k=="atomic_add";
            if (k != "read" && k != "write")
              return;
            auto ty = cast<MemRefType>(i->getOperand(0).getType());
            double bytes = m * n * ty.getElementTypeBitWidth() / 8.;
            if (k == "read")
              r += bytes;
            else if (i->hasAttr("axes") != swapped)
              ws += bytes;
            else
              w += bytes;
          });
          r *= stream ? 1 + reductions : 1;
          auto [keys,segmentedRead]=intentdsl::kir::valueTraffic(op,*f);
          r+=segmentedRead;
          r+=4*m*n*atomics;ws+=4*m*n*atomics;
          r+=keys;
          if(segmentedRead)e.terms["packed_member_reads"]={segmentedRead,"B"};
          auto capacityFacts =
              ctx.target.services.get<intentdsl::RowCapacity>();
          auto sector = capacityFacts ? capacityFacts->storeSectorBytes : 0;
          const bool permutation =
              choice.get("walk") && intentdsl::rectangularCopy(op);
          double live=0, peak=0;
          for(auto &i:op->getRegion(0).front()){
            for(auto v:i.getResults()) if(isa<RankedTensorType>(v.getType()) && !v.use_empty())live++;
            peak=std::max(peak,live);
            llvm::SmallDenseSet<Value> consumed;
            for(auto v:i.getOperands())
              if(v.getDefiningOp() && isa<RankedTensorType>(v.getType()) && consumed.insert(v).second &&
                 llvm::none_of(v.getUsers(),[&](Operation *u){return u->getBlock()==i.getBlock() && i.isBeforeInBlock(u);}))live--;
          }
          auto regFile=ctx.target.services.facts.find("register_file")?intentdsl::quantity(ctx,"register_file","reg32"):std::nullopt;
          // A rectangular projected input can issue one pitched read for the
          // entire row group. Charge the setup once per participating wave.
          std::optional<double> dmaSetup;
          unsigned rectangles = 0;
          if (!stream && ctx.target.services.facts.find("dma_in_latency")) {
            auto fn = op->getParentOfType<func::FuncOp>();
            llvm::SmallDenseSet<Value> seen;
            op->walk([&](Operation *readOp) {
              if (readOp->getName().getStringRef() != "block.read") return;
              auto slot = dyn_cast<BlockArgument>(readOp->getOperand(0));
              if (!slot || slot.getOwner() != &fn.front() || !seen.insert(slot).second) return;
              auto attr = fn.getArgAttrOfType<AffineMapAttr>(slot.getArgNumber(), "kir.indexing");
              if (!attr) return;
              auto map = attr.getValue();
              if (map.getNumDims() < 3 || map.getNumResults() < 2) return;
              auto inner = dyn_cast<AffineDimExpr>(map.getResults().back());
              if (inner && inner.getPosition()+1 == map.getNumDims() &&
                  map.getResult(map.getNumResults()-2).isFunctionOfDim(map.getNumDims()-2))
                ++rectangles;
            });
            if (rectangles) dmaSetup=intentdsl::quantity(ctx,"dma_in_latency","ns");
          }
          for (auto value : choice.getAs<ArrayAttr>("rows")) {
            auto row = cast<DictionaryAttr>(value);
            double rows = row.getAs<IntegerAttr>("rows").getInt(),
                   width = row.getAs<IntegerAttr>("width").getInt();
            double instances = std::ceil(m / rows),
                   tiles = std::ceil(n / width),
                   waves = std::ceil(instances / f->participants);
            auto &v =
                e.rows[std::to_string(row.getAs<IntegerAttr>("id").getInt())];
            v.terms = {
                {"read_bytes", {r, "B"}},
                {"regular_store", {w, "B"}},
                {"strided_store", {ws, "B"}},
                {"waves", {waves, "count"}},
                {"tail", {instances * rows * tiles * width - m * n, "lane"}}};
            auto walk=choice.getAs<StringAttr>("walk");
            bool gridTiles=walk && (walk=="grid-tile" || walk=="segment-tile" || walk=="column-panels");
            // Independent columns create concurrently schedulable instances.
            // The other streaming walks issue all column tiles serially.
            double scheduled=gridTiles ? instances*tiles : instances;
            double serial=gridTiles ? 1 : tiles;
            v.terms["serial_tiles"]={serial,"count"};
            double regs=(peak+summaryFields)*rows*width;
            if (summaryFields) v.terms["summary_live_words"]={double(summaryFields*rows*width),"reg32"};
            double resident=regFile && regs>0?std::max(1.,std::floor(*regFile/regs)):1.;
            v.terms["fragment_register_estimate"]={regs,"reg32"};
            v.terms["resident_instance_upper_estimate"]={resident,"instance"};
            v.terms["active_instances"]={std::min(instances,resident*f->participants),"count"};
            double active = std::min(scheduled, resident * f->participants);
            v.terms["active_instances"]={active,"count"};
            double rounds = active ? std::ceil(scheduled / active) : 0.;
            double capacity = resident * f->participants;
            double laneEfficiency = instances && tiles ? (m*n)/(instances*rows*tiles*width) : 1.;
            // Each memory warp still issues a transaction for its active tail.
            // Count sectors, not bytes for masked-off lanes.
            double sectors = std::ceil(n/16.);
            double allocatedSectors = tiles * std::ceil(width/16.);
            double tailCost = laneEfficiency ? std::min(1./laneEfficiency,
                sectors ? allocatedSectors/sectors : 1.) : 1.;
            v.terms["lane_efficiency"]={laneEfficiency,"ratio"};
            v.terms["resident_waves"]={rounds,"count"};
            double scanSteps=scans*rounds*std::ceil(std::log2(width));
            v.terms["prefix_exchange_steps"]={scanSteps,"count"};
            if(atomics)v.terms["atomic_updates"]={m*n*atomics,"count"};
            if(keys)v.terms["key_input_bytes"]={keys,"B"};
            if (read && write && strided && stridedRead && launch) {
              double scatterRate = *strided;
              if (permutation && sector && ws) {
                // Adjacent source rows write adjacent destination columns.
                // Each packet shares a sector up to its physical byte width.
                double bytes = ws / std::max(1., m * n);
                double neighbors = std::min(rows, std::max(1., sector / bytes));
                scatterRate = std::min(*write, *strided * neighbors);
                v.terms["contiguous_store_neighbors"] = {neighbors, "count"};
              }
              v.timeNs =
                  *launch +
                  (r / (swapped ? *stridedRead : *read) + w / *write +
                   ws / scatterRate) *
                      (scheduled ? rounds *
                                       std::min(capacity,
                                                double(f->participants) * 2) /
                                       scheduled
                                 : 0.) *
                      tailCost;
              if (dmaSetup) {
                double submission = waves * rectangles * *dmaSetup;
                v.terms["projected_dma_setup"] = {submission, "ns"};
                *v.timeNs += submission;
              }
              if(auto mode=choice.getAs<StringAttr>("bins")){
                auto caps=ctx.target.services.get<intentdsl::RowCapacity>();
                double updates=0;
                op->walk([&](Operation *k){if(k->getName().stripDialect()=="count_keys")
                  updates+=f->viewBytes(k->getOperand(0))/(cast<MemRefType>(k->getOperand(0).getType()).getElementTypeBitWidth()/8.);});
                // With no distribution promise, estimate contention for
                // uniform keys over the active bins in each partition.
                double perBin=updates/std::max(1.,m*n);
                double collisions=perBin>0?updates-m*n*(-std::expm1(-perBin)):0.;
                bool local=mode=="private-merge";
                v.terms["private_bin_bytes"]={local?4.*width:0.,"B"};
                if(caps->privateBinCapacity)v.terms["per_block_storage_limit"]={double(*caps->privateBinCapacity),"B"};
                v.terms["bin_merge_bytes"]={local?4.*m*n:0.,"B"};
                v.terms["contending_global_updates"]={local?0.:collisions,"count"};
                if(local){
                  auto rate=intentdsl::quantity(ctx,"compute","lane/ns");
                  if(rate)*v.timeNs+=updates*(caps->nativePrivateHistogram?1.:width)/ *rate;
                  if(!caps->nativePrivateHistogram){
                    // Compare/reduce materializes one i32 predicate per key
                    // and bin inside the tile. Use on-chip traffic as a proxy
                    // when the target has no collective instruction.
                    auto localRate=intentdsl::quantity(ctx,"read_1_shared_bw","B/ns").value_or(*read);
                    *v.timeNs+=4.*updates*width/localRate;
                    v.terms["local_predicate_bytes"]={4.*updates*width,"B"};
                  }
                  *v.timeNs+=4.*m*n/ *write;
                }else *v.timeNs+=8.*updates/ *strided+4.*collisions/ *write;
              }
              if(scanSteps){
                auto rate=intentdsl::quantity(ctx,"compute","lane/ns");
                if(rate)*v.timeNs+=scanSteps*rows*width / *rate;
              }
              if(segmentedRead && gridTiles && stridedRead){
                // Reuse the same offset endpoints across a group of independent
                // member loads. Preserve the serial additions, which still cost
                // one operation per member.
                auto batch=choice.getAs<IntegerAttr>("member_batch");
                double groups=std::ceil(segmentedRead/std::max(1.,m*n*4.)/(batch?batch.getInt():1));
                v.terms["dependent_load_groups"]={groups,"count"};
                *v.timeNs+=groups*rounds*128.*f->participants/ *stridedRead;
              }
              if (!e.timeNs || *v.timeNs < *e.timeNs) {
                e.timeNs = v.timeNs;
                e.terms = v.terms;
              }
            } else
              e.terms = v.terms;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
