#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
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
          unsigned reductions = 0;
          op->walk([&](Operation *i) {
            auto k = i->getName().stripDialect();
            reductions += k == "sum" || k == "max_num";
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
            double regs=peak*rows*width;
            double resident=regFile && regs>0?std::max(1.,std::floor(*regFile/regs)):1.;
            v.terms["fragment_register_estimate"]={regs,"reg32"};
            v.terms["resident_instance_upper_estimate"]={resident,"instance"};
            v.terms["active_instances"]={std::min(instances,resident*f->participants),"count"};
            double occupancyPenalty=regFile?2./std::min(2.,resident):1.;
            if (read && write && strided && stridedRead && launch) {
              v.timeNs =
                  *launch +
                  (r / (swapped ? *stridedRead : *read) + w / *write +
                   ws / *strided) *
                      (instances ? waves * f->participants / instances : 0.) * occupancyPenalty;
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
