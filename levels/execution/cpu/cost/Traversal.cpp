#include "intentdsl/levels/algorithm/kir/ir/Traffic.h"
#include "intentdsl/levels/algorithm/kir/ir/Permutation.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
CostEstimate vectorCost(Operation *, const StructuralCandidate &, CostContext &);
// Interpolate inverse throughput: both axes describe observed memory access
// geometry. Padded and coprime pitches have different cache set reuse.
static std::optional<double> copyTime(CostContext &ctx, double bytes,
                                      double stride, double span, bool gather) {
  bool aligned = std::fmod(stride, 32.) == 0;
  int base = stride <= 128 ? 32 : 128;
  int p0 = base - !aligned, p1 = base * 4 - !aligned;
  int n0 = span <= 256 ? 32 : 256, n1 = span <= 256 ? 256 : 1024;
  double x=std::clamp((stride-p0)/(p1-p0),0.,1.);
  double y=std::clamp((span-n0)/(n1-n0),0.,1.);
  std::string prefix=gather ? "span_gather_" : "span_scatter_";
  double time=0;
  for(auto [pitch,wx]:{std::pair{p0,1-x},std::pair{p1,x}})
    for(auto [length,wy]:{std::pair{n0,1-y},std::pair{n1,y}}) {
      auto rate=intentdsl::quantity(ctx,prefix+std::to_string(pitch)+"_"+
                                    std::to_string(length)+"_bw","B/ns");
      if(!rate || *rate<=0) return std::nullopt;
      time+=wx*wy*bytes/ *rate;
    }
  return time;
}
void registerCost(Unit &u) {
  for (bool permuted : {false, true})
    u.costModels.push_back(
        {{permuted ? "permutation-time" : "task-time",
          u.qualify(permuted ? "permutation" : "partition"), "1",
          [](Operation *op, const StructuralCandidate &c,
             CostContext &ctx) -> llvm::Expected<CostEstimate> {
            auto *facts = ctx.target.services.get<intentdsl::CallFacts>();
            auto [m, n] = facts->geometry(op);
            auto d = cast<DictionaryAttr>(c.value);
            auto walk = d.getAs<StringAttr>("walk");
            auto rows = d.getAs<ArrayAttr>("rows");
            CostEstimate e = vectorCost(op, c, ctx);
            e.scope = "full_call";
            e.bottleneck = "memory traffic";
            auto rd = intentdsl::quantity(ctx, "read_1", "B/ns"),
                 wr = intentdsl::quantity(ctx, "write_1", "B/ns");
            auto rs = intentdsl::quantity(ctx, "read_16", "B/ns"),
                 ws = intentdsl::quantity(ctx, "write_16", "B/ns");
            auto launch = intentdsl::quantity(ctx, "launch", "ns"),
                 sync = intentdsl::quantity(ctx, "sync", "ns");
            double read = 0, write = 0, readNs = 0, writeNs = 0, lines = 0;
            double readStep = 1, writeStep = 1;
            llvm::Error strideError=llvm::Error::success();
            op->walk([&](Operation *item) {
              if(strideError)return;
              auto k = item->getName().stripDialect();
              if (k != "read" && k != "write")
                return;
              auto arg = dyn_cast<BlockArgument>(item->getOperand(0));
              if (!arg)
                return;
              auto v = cast<MemRefType>(arg.getType());
              auto bytes = v.getElementTypeBitWidth() / 8;
              bool reversed = item->hasAttr("axes") != (walk == "destination");
              auto axis = v.getRank() == 2 && reversed ? 0 : v.getRank() - 1;
              auto stride=facts->axisStride(item,arg.getArgNumber(),axis);
              if(!stride){strideError=stride.takeError();return;}
              double step=std::abs(*stride);
              double mix = std::clamp((step - 1) / 15., 0., 1.);
              // A rectangle bounds the iteration count; its inner loop still
              // addresses the same axis as the source-row implementation.
              if (k == "read")
                readStep = step;
              else
                writeStep = step;
              double traffic = m * n * bytes;
              double effStep =
                  walk == "tile" ? 1. : std::min(std::max(step, 1.), 16.);
              lines += std::ceil(traffic * effStep / 64.);
              if (k == "read") {
                read += traffic;
                if (rd && rs)
                  readNs += traffic * ((1 - mix) / *rd + mix / *rs);
              } else {
                write += traffic;
                if (wr && ws)
                  writeNs += traffic * ((1 - mix) / *wr + mix / *ws);
              }
            });
            if(strideError)return std::move(strideError);
            auto [keys,segments]=intentdsl::kir::valueTraffic(op,*facts);
            if(keys){read+=keys;if(rd)readNs+=keys/ *rd;e.terms["key_input_bytes"]={keys,"B"};}
            if(segments){read+=segments;if(rd)readNs+=segments/ *rd;e.terms["packed_member_reads"]={segments,"B"};}
            e.terms["read"] = {read, "B"};
            e.terms["write"] = {write, "B"};
            if (intentdsl::rectangularCopy(op) && read == write &&
                std::min(readStep, writeStep) == 1) {
              bool gather = readStep > writeStep;
              double extent=walk=="destination" ? m : n;
              double span=walk=="tile" ? std::min(n,32.) : extent;
              e.terms["inner_span"]={span,"element"};
              std::optional<double> time;
              const double pitch=std::max(readStep,writeStep);
              const double owners=walk=="tile" ? std::ceil(m/32)*std::ceil(n/32) : (walk=="destination"?n:m);
              if (std::min(owners, double(facts->participants))>1) {
                // Concurrent row owners advance adjacent stripes of one bank.
                // Price both streams from the shared-bank experiment; the
                // private span probe has no competing writer for its cache lines.
                int low=pitch<=128?32:128, high=4*low;
                std::string prefix=gather?"panel_gather_":"panel_scatter_";
                auto a=intentdsl::quantity(ctx,prefix+std::to_string(low)+"_bw","B/ns");
                auto b=intentdsl::quantity(ctx,prefix+std::to_string(high)+"_bw","B/ns");
                if(a && b){
                  auto t=std::clamp((pitch-low)/(high-low),0.,1.);
                  time=read*((1-t)/ *a+t/ *b);
                }
                e.terms["concurrent_owners"]={std::min(owners,double(facts->participants)),"count"};
              } else time=copyTime(ctx,read,pitch,span,gather);
              if (time) {
                readNs = *time;
                writeNs = 0;
                e.terms["paired_copy"] = {readNs, "ns"};
              }
            }
            e.terms["inner_read_stride"] = {readStep, "element"};
            e.terms["inner_write_stride"] = {writeStep, "element"};
            e.terms["line_touches"] = {lines, "line"};
            auto arithmetic = e.timeNs;
            double sharedLines=0;
            op->walk([&](Operation *update){
              if(update->getName().stripDialect()!="atomic_add")return;
              auto index=cast<BlockArgument>(update->getOperand(0)).getArgNumber();
              auto *arg=facts->arguments[index].getAsObject();auto shape=arg->getArray("shape");
              if(shape && shape->size()==1)sharedLines=std::max(1.,std::ceil(shape->front().getAsInteger().value_or(0)*4./64));
            });
            e.timeNs.reset();
            for (auto [i, row] : llvm::enumerate(rows)) {
              auto grain = cast<DictionaryAttr>(row)
                               .getAs<IntegerAttr>("grain")
                               .getInt();
              double groups =
                  walk == "tile"
                      ? std::ceil(n / 32.) * std::ceil(m / 32.)
                      : std::ceil(((walk == "destination" ||
                                    d.contains("member_tasks"))
                                       ? n
                                       : m) /
                                  grain);
              double active = std::min(groups, double(facts->participants));
              RowPrediction pred;
              pred.terms = e.terms;
              pred.terms["tasks"] = {groups, "count"};
              pred.terms["active_workers"] = {active, "count"};
              if (rd && wr && rs && ws && launch && sync && arithmetic) {
                double work = std::max(readNs + writeNs, *arithmetic);
                pred.timeNs =
                    *launch + *sync +
                    work * (groups ? std::ceil(groups / facts->participants) *
                                         facts->participants / groups
                                   : 0.);
                if(sharedLines){
                  bool sharded=d.contains("key_extent");
                  double writers=d.contains("single_dispatcher")?1:sharded?std::min(sharedLines,double(facts->participants)):std::min(groups,double(facts->participants));
                  double collisions=sharded?0:std::max(0.,writers-1)/sharedLines;
                  pred.terms["shared_cacheline_writers"]={collisions,"ratio"};
                  if(rs && rd){
                    auto traffic=m*n*(64/std::max(1.,writers)/ *rs+64*collisions/ *rs);
                    if(sharded)traffic+=read*(writers-1)/ *rd;
                    if(auto bins=d.getAs<IntegerAttr>("batch_extent")){
                      double reservations=std::min(m*n,writers*bins.getInt());
                      pred.terms["maximum_reservations"]={reservations,"count"};
                      traffic=reservations*(64/std::max(1.,writers)+64*collisions)/ *rs+read/ *rd;
                    }
                    *pred.timeNs+=traffic;
                  }
                }
                if (!e.timeNs || *pred.timeNs < *e.timeNs)
                  e.timeNs = pred.timeNs;
              } else {
                double work = std::max(lines * 4.0, arithmetic.value_or(0.0));
                pred.timeNs =
                    200.0 +
                    work * (groups ? std::ceil(groups / facts->participants) *
                                         facts->participants / groups
                                   : 0.);
                if (!e.timeNs || *pred.timeNs < *e.timeNs)
                  e.timeNs = pred.timeNs;
              }
              e.rows[std::to_string(i)] = std::move(pred);
            }
            return e;
          }},
         COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
