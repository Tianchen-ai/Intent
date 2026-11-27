#include "intentdsl/levels/execution/block/ir/Traffic.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void supplyCost(Unit &u) {
  u.costModels.push_back(
      {{"pipeline-time", u.qualify("supply"), "3",
        [](Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto d = cast<DictionaryAttr>(c.value);
          auto *caps = ctx.target.services.get<intentdsl::dsa::LocalSupply>();
          if (d.contains("library_impl")) {
            if (!caps || caps->libraryEstimator.empty())
              return llvm::createStringError(
                  "missing target library supply estimator");
            return ctx.local(caps->libraryEstimator);
          }
          auto *facts = ctx.target.services.get<intentdsl::CallFacts>();
          auto [m, n] = facts->geometry(op);
          auto dma = ctx.local(facts->localEstimator);
          if (!dma)
            return dma.takeError();
          auto e = *dma;
          e.scope = "full_call";
          e.bottleneck = "supply/compute";
          auto sync = intentdsl::quantity(ctx, "sync", "ns"),
               launch = intentdsl::quantity(ctx, "launch", "ns"),
               vector = intentdsl::quantity(ctx, "compute", "lane/ns");
          auto startup = intentdsl::quantity(ctx, "first_io", "ns");
          auto optQty = [&](llvm::StringRef name,
                            llvm::StringRef unit) -> std::optional<double> {
            if (!ctx.target.services.facts.find(name))
              return std::nullopt;
            return intentdsl::quantity(ctx, name, unit);
          };
          auto rtIn = optQty("dma_in_sync_latency", "ns"),
               rtOut = optQty("dma_out_sync_latency", "ns"),
               singleInBw = optQty("dma_in_single_bw", "B/ns"),
               singleOutBw = optQty("dma_out_single_bw", "B/ns"),
               fullInBw = optQty("dma_in_bw", "B/ns"),
               fullOutBw = optQty("dma_out_bw", "B/ns"),
               inLat = optQty("dma_in_latency", "ns"),
               outLat = optQty("dma_out_latency", "ns"),
               scatterBw = optQty("write_16", "B/ns");
          double maxConsumers = caps && caps->consumers > 0
                                    ? double(caps->consumers)
                                    : 16.;
          auto [keys, segmentWork] = intentdsl::block::realizedTraffic(op, *facts);
          auto depth = d.getAs<IntegerAttr>("depth").getInt();
          double participants = d.getAs<IntegerAttr>("participants").getInt();
          double batch = d.getAs<IntegerAttr>("batch")
                             ? d.getAs<IntegerAttr>("batch").getInt()
                             : 1.;
          auto configs = op->getAttrOfType<ArrayAttr>("block.configs");
          unsigned vecOps = 0, scalarOps = 0;
          unsigned summaries = 0, scans = 0, atomics = 0;
          bool hasIndexed = false;
          double indirectMembers = 0.;
          int64_t keySpan = 0;
          op->walk([&](Operation *i) {
            auto k = i->getName().stripDialect();
            summaries += k == "summary_reduce";
            scans += k == "scan_sum";
            if (k == "key_counts")
              keySpan = i->getAttrOfType<IntegerAttr>("span").getInt();
            atomics += k == "atomic_add_at";
            hasIndexed |= k == "indexed_tile_read";
            if (k == "member_load" && i->getNumOperands() == 2) {
              auto *index = i->getOperand(1).getDefiningOp();
              if (index && index->getName().stripDialect() == "member_load")
                indirectMembers +=
                    facts->viewBytes(index->getOperand(0)) /
                    (cast<MemRefType>(index->getOperand(0).getType())
                         .getElementTypeBitWidth() /
                     8.);
            }
            if (k == "reduce") {
              ++vecOps;
              return;
            }
            bool isArith = k == "add" || k == "sub" || k == "mul" ||
                           k == "div" || k == "exp" || k == "exp2" ||
                           k == "rsqrt" || k == "sqrt" || k == "maximum" ||
                           k == "minimum" || k == "select" || k == "cast";
            if (isArith && i->getNumResults() == 1) {
              if (auto rTy =
                      dyn_cast<RankedTensorType>(i->getResult(0).getType());
                  rTy && rTy.getRank() >= 2)
                ++vecOps;
              else
                ++scalarOps;
            }
          });
          bool hasDetailedTransfer =
              rtIn && rtOut && fullInBw && fullOutBw && inLat && outLat &&
              scatterBw && e.terms.count("step_in_bytes") &&
              e.terms.count("step_out_bytes") &&
              e.terms.count("step_scatter_bytes") &&
              e.terms.count("shared_in_bytes") &&
              e.terms.count("shared_in_requests") &&
              e.terms.count("grouped_in_requests") &&
              e.terms.count("per_row_in_requests") &&
              e.terms.count("grouped_out_requests") &&
              e.terms.count("per_row_out_requests");
          e.timeNs.reset();
          for (auto a : d.getAs<ArrayAttr>("rows")) {
            auto row = cast<DictionaryAttr>(a);
            auto id = row.get("id");
            auto it = llvm::find_if(configs, [&](Attribute v) {
              return cast<DictionaryAttr>(v).get("id") == id;
            });
            double group =
                cast<DictionaryAttr>(*it).getAs<IntegerAttr>("rows").getInt();
            double effGroup = group * batch;
            double groupsTotal = std::ceil(m / effGroup);
            double tileFraction = 1.;
            if (d.contains("independent_columns")) {
              auto width =
                  cast<DictionaryAttr>(*it).getAs<IntegerAttr>("width").getInt();
              groupsTotal *= std::ceil(n / width);
              tileFraction =
                  std::min(1., double(width) / std::max(1., double(n)));
            }
            double steps = std::ceil(groupsTotal / participants);
            double activeParticipants =
                std::min(participants, std::max(1., groupsTotal));
            RowPrediction pred;
            pred.terms = dma->terms;
            pred.terms["steps"] = {steps, "count"};
            if (segmentWork)
              pred.terms["packed_member_reads"] = {segmentWork, "B"};
            if (keys)
              pred.terms["key_input_bytes"] = {keys, "B"};
            double merges = effGroup * std::max(0., n - 1.) * summaries;
            pred.terms["summary_merges_per_group"] = {merges, "count"};
            pred.terms["serial_prefix_adds"] = {
                effGroup * std::max(0., n - 1.) * scans, "count"};
            if (atomics)
              pred.terms["atomic_updates_per_group"] = {
                  effGroup * n * atomics, "count"};
            pred.terms["working_set"] = {
                double(row.getAs<IntegerAttr>("bytes").getInt()), "B"};
            if (dma->timeNs && launch && sync && vector && startup) {
              double fetch = 0., write = 0., sharedPrologue = 0., firstIo = 0.;
              if (hasDetailedTransfer) {
                double bwIn =
                    singleInBw
                        ? std::min(*singleInBw,
                                   (maxConsumers * (*fullInBw)) /
                                       activeParticipants)
                        : *fullInBw;
                double bwOut =
                    singleOutBw
                        ? std::min(*singleOutBw,
                                   (maxConsumers * (*fullOutBw)) /
                                       activeParticipants)
                        : *fullOutBw;
                double stepInB = e.terms.at("step_in_bytes").value;
                double stepOutB = e.terms.at("step_out_bytes").value;
                double stepScatB = e.terms.at("step_scatter_bytes").value;
                double gInR = e.terms.at("grouped_in_requests").value;
                double rInR = e.terms.at("per_row_in_requests").value;
                double gOutR = e.terms.at("grouped_out_requests").value;
                double rOutR = e.terms.at("per_row_out_requests").value;
                bool hasStepIn = stepInB > 0. || gInR > 0. || rInR > 0.;
                bool hasStepOut =
                    stepOutB > 0. || stepScatB > 0. || gOutR > 0. || rOutR > 0.;
                fetch = (hasStepIn ? *rtIn : 0.) +
                        effGroup * tileFraction * (stepInB / bwIn) +
                        (gInR + effGroup * rInR) * (*inLat);
                write = (hasStepOut ? *rtOut : 0.) +
                        effGroup * tileFraction *
                            (stepOutB / bwOut + stepScatB / (*scatterBw)) +
                        (gOutR + effGroup * rOutR) * (*outLat);
                double sharedInB = e.terms.at("shared_in_bytes").value;
                double sharedInR = e.terms.at("shared_in_requests").value;
                if (steps > 0. && (sharedInB > 0. || sharedInR > 0.))
                  sharedPrologue = sharedInB / bwIn + sharedInR * (*inLat);
              } else {
                fetch = effGroup * tileFraction * e.terms.at("fetch").value;
                write = effGroup * tileFraction * e.terms.at("writeback").value;
                firstIo = steps ? *startup : 0.;
              }
              double compute =
                  tileFraction *
                  ((effGroup * n *
                        std::max(0., double(vecOps) -
                                         e.terms.at("native_operations").value) +
                    effGroup * double(scalarOps) + 8 * merges +
                    pred.terms["serial_prefix_adds"].value) /
                       *vector +
                   effGroup * e.terms.at("native_compute").value);
              if (keySpan) {
                auto panel = d.getAs<DenseI64ArrayAttr>("histogram_panel");
                double work = effGroup * keySpan;
                if (panel) {
                  double packets = std::ceil(panel[0] / 32.0);
                  work = effGroup * packets * (1 + panel[1] / 64) +
                         panel[1] / std::max(steps, 1.0);
                }
                pred.terms["key_count_operations"] = {work, "count"};
                compute += work / *vector;
              }
              double overlap =
                  depth == 2
                      ? std::max(0., steps - 1) *
                            (hasIndexed && d.contains("async_writeback")
                                 ? std::max(std::min(fetch, compute),
                                            0.25 * std::min(fetch, write))
                                 : std::min(fetch, compute))
                      : 0.;
              double waitNs =
                  (depth == 2 || hasIndexed) ? steps * *sync : 0.;
              pred.terms["overlapped"] = {overlap, "ns"};
              pred.terms["fill"] = {fetch, "ns"};
              pred.terms["drain"] = {compute + write, "ns"};
              pred.terms["launch"] = {*launch, "ns"};
              pred.terms["first_io_excess"] = {firstIo, "ns"};
              if (sharedPrologue > 0.)
                pred.terms["shared_prologue"] = {sharedPrologue, "ns"};
              pred.terms["wait"] = {waitNs, "ns"};
              pred.timeNs = *launch + firstIo + sharedPrologue +
                            steps * (fetch + compute + write) - overlap +
                            waitNs;
              if (atomics) {
                double members = effGroup * n;
                auto ticket = d.getAs<DenseI64ArrayAttr>("ticket_panel");
                double issues =
                    ticket ? std::min<double>(ticket[0], members) : members;
                *pred.timeNs += steps * atomics * issues * (*sync);
                if (ticket)
                  *pred.timeNs += steps * (members + ticket[0]) / (*vector);
                pred.terms["atomic_reservations"] = {steps * atomics * issues,
                                                     "count"};
              }
              if(segmentWork || keys){
                auto rate=intentdsl::quantity(ctx,"read_1","B/ns");
                if (rate) {
                  auto gather = d.getAs<DenseI64ArrayAttr>("segment_gather");
                  double traffic = (segmentWork + keys) / *rate;
                  if (gather && fullInBw) {
                    // Stream packed members; each participant stages its
                    // lookup table once before processing independent tiles.
                    double bandwidth = std::min(singleInBw.value_or(*fullInBw) *
                                                    activeParticipants,
                                                *fullInBw * maxConsumers);
                    traffic =
                        (segmentWork + 4.0 * gather[0] * activeParticipants) /
                        bandwidth;
                    // One dependent scalar addition for each live member.
                    traffic += segmentWork / (3 * sizeof(float) *
                                              activeParticipants * (*vector));
                  }
                  if (!gather && inLat)
                    traffic += indirectMembers * (*inLat) / activeParticipants;
                  *pred.timeNs += traffic;
                }
              }
              if (!e.timeNs || *pred.timeNs < *e.timeNs)
                e.timeNs = pred.timeNs;
            }
            e.rows[std::to_string(cast<IntegerAttr>(id).getInt())] =
                std::move(pred);
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
