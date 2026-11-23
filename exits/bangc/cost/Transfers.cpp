#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void dmaCost(Unit &u) {
  u.costModels.push_back(
      {{"dma", "levels/storage/dsa::supply", "1",
        [](Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          CostEstimate e;
          e.scope = "one row DMA";
          e.bottleneck = "transfer";
          auto [m, n] =
              ctx.target.services.get<intentdsl::CallFacts>()->geometry(op);
          auto choice = cast<DictionaryAttr>(c.value);
          double batch = choice.getAs<IntegerAttr>("batch")
                             ? choice.getAs<IntegerAttr>("batch").getInt()
                             : 1.;
          bool canTileTranspose = false;
          if (auto cfgs = op->getAttrOfType<ArrayAttr>("block.configs");
              cfgs && choice.getAs<ArrayAttr>("rows")) {
            for (auto a : choice.getAs<ArrayAttr>("rows")) {
              auto id = cast<DictionaryAttr>(a).get("id");
              for (auto cfg : cfgs) {
                auto row = cast<DictionaryAttr>(cfg);
                if (row.get("id") == id) {
                  int64_t effR =
                      row.getAs<IntegerAttr>("rows").getInt() * int64_t(batch);
                  if (effR >= 32 && (effR % 32) == 0)
                    canTileTranspose = true;
                }
              }
            }
          }
          double in = 0, out = 0, r = 0, w = 0, scatter = 0;
          op->walk([&](Operation *i) {
            auto k = i->getName().stripDialect();
            if (k != "load" && k != "store")
              return;
            auto view = cast<MemRefType>(i->getOperand(0).getType());
            double bytes = n * view.getElementTypeBitWidth() / 8;
            if (k == "load") {
              double frequency = 1.;
              if (choice.getAs<BoolAttr>("persistent").getValue() &&
                  view.getRank() == 1) {
                frequency =
                    1. /
                    std::max(
                        1.,
                        std::ceil(m / choice.getAs<IntegerAttr>("participants")
                                          .getInt()));
                in += frequency * bytes;
                r += frequency;
              } else {
                in += bytes;
                r += view.getRank() >= 2 ? 1. / batch : 1.;
              }
            } else {
              out += bytes;
              bool permuted = false;
              if (auto axes =
                      i->getAttrOfType<DenseI64ArrayAttr>("block.member_axes");
                  axes && axes.size() == 2 && axes[0] == 1) {
                if (!canTileTranspose) {
                  scatter += bytes;
                  permuted = true;
                }
              }
              w += (!permuted && view.getRank() >= 2) ? 1. / batch : 1.;
            }
          });
          auto bi = intentdsl::quantity(ctx, "dma_in_bw", "B/ns"),
               bo = intentdsl::quantity(ctx, "dma_out_bw", "B/ns"),
               li = intentdsl::quantity(ctx, "dma_in_latency", "ns"),
               lo = intentdsl::quantity(ctx, "dma_out_latency", "ns"),
               bw = intentdsl::quantity(ctx, "write_16", "B/ns");
          e.terms = {{"incoming", {in, "B"}},
                     {"outgoing", {out, "B"}},
                     {"requests", {r + w, "count"}}};
          double specialized = 0, count = 0;
          double lanes = n * batch;
          op->walk([&](Operation *i) {
            auto code=i->getName().stripDialect();
            std::string primitive;
            if(code=="reduce"){
              auto algebra=i->getAttrOfType<StringAttr>("algebra");
              if(algebra=="max_num")primitive="max";
              if(algebra=="sum")primitive="sum";
            } else if (code=="exp") primitive="exp";
            else if (code=="div" && i->hasOneUse()){
              Operation *consumer=*i->getUsers().begin();
              auto ty=consumer->getNumResults()==1?dyn_cast<ShapedType>(consumer->getResult(0).getType()):ShapedType{};
              if(consumer->getName().stripDialect()=="cast" && ty && ty.getElementType().isF16())
                primitive="quotient";
            }
            if(primitive.empty())return;
            int low = lanes <= 512 ? 64 : 512, high = lanes <= 512 ? 512 : 2048;
            auto prefix="compute_"+primitive+"_";
            auto a=intentdsl::quantity(ctx,prefix+std::to_string(low),"ns");
            auto b=intentdsl::quantity(ctx,prefix+std::to_string(high),"ns");
            if(a && b){
              double u = lanes <= 512
                             ? std::clamp((lanes - low) / (high - low), 0., 1.)
                             : (lanes - low) / (high - low);
              specialized += ((*a) + ((*b) - (*a)) * u) / batch;
              ++count;
            }
          });
          e.terms["native_compute"]={specialized,"ns"};
          e.terms["native_operations"]={count,"count"};
          if (bi && bo && li && lo && bw) {
            double fetch = in / *bi + r * *li,
                   drain = (out - scatter) / *bo + scatter / *bw + w * *lo;
            e.terms["fetch"] = {fetch, "ns"};
            e.terms["writeback"] = {drain, "ns"};
            e.timeNs = fetch + drain;
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
