#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "../capabilities/VectorMath.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void dmaCost(Unit &u) {
  u.costModels.push_back(
      {{"dma", "levels/storage/dsa::supply", "3",
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
          double stepIn = 0, sharedIn = 0, sharedReqs = 0;
          double groupedInReqs = 0, perRowInReqs = 0;
          double groupedOutReqs = 0, perRowOutReqs = 0;
          op->walk([&](Operation *i) {
            auto k = i->getName().stripDialect();
            if (k != "load" && k != "store" && k != "indexed_tile_read")
              return;
            auto view = cast<MemRefType>(i->getOperand(0).getType());
            double bytes = n * view.getElementTypeBitWidth() / 8;
            if (k == "indexed_tile_read") {
              in += bytes + 8.;
              stepIn += bytes + 8.;
              r += 1.;
              perRowInReqs += 1.;
            } else if (k == "load") {
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
                sharedIn += bytes;
                sharedReqs += 1.;
              } else {
                in += bytes;
                stepIn += bytes;
                if (view.getRank() >= 2) {
                  r += 1. / batch;
                  groupedInReqs += 1.;
                } else {
                  r += 1.;
                  perRowInReqs += 1.;
                }
              }
            } else {
              out += bytes;
              bool permuted = false;
              if (auto axes =
                      i->getAttrOfType<DenseI64ArrayAttr>("block.member_axes");
                  axes && axes.size() == 2 && axes[0] == 1) {
                scatter += canTileTranspose ? 0.85 * bytes : bytes;
                permuted = !canTileTranspose;
              }
              if (!permuted && view.getRank() >= 2) {
                w += 1. / batch;
                groupedOutReqs += 1.;
              } else {
                w += 1.;
                perRowOutReqs += 1.;
              }
            }
          });
          auto bi = intentdsl::quantity(ctx, "dma_in_bw", "B/ns"),
               bo = intentdsl::quantity(ctx, "dma_out_bw", "B/ns"),
               li = intentdsl::quantity(ctx, "dma_in_latency", "ns"),
               lo = intentdsl::quantity(ctx, "dma_out_latency", "ns"),
               bw = intentdsl::quantity(ctx, "write_16", "B/ns");
          e.terms = {{"incoming", {in, "B"}},
                     {"outgoing", {out, "B"}},
                     {"requests", {r + w, "count"}},
                     {"step_in_bytes", {stepIn, "B"}},
                     {"step_out_bytes", {out - scatter, "B"}},
                     {"step_scatter_bytes", {scatter, "B"}},
                     {"shared_in_bytes", {sharedIn, "B"}},
                     {"shared_in_requests", {sharedReqs, "count"}},
                     {"grouped_in_requests", {groupedInReqs, "count"}},
                     {"per_row_in_requests", {perRowInReqs, "count"}},
                     {"grouped_out_requests", {groupedOutReqs, "count"}},
                     {"per_row_out_requests", {perRowOutReqs, "count"}}};
          double specialized = 0, count = 0;
          double mathPasses = 0;
          double lanes = n * batch;
          op->walk([&](Operation *i) {
            auto code=i->getName().stripDialect();
            if (approximateScratch(i) &&
                (code == "sqrt" || choice.contains("approximate_math"))) {
              // Strict sqrt certifies candidates; reciprocal and fast sqrt use
              // only the vector traversals permitted by the effective IR flags.
              bool fast = choice.contains("approximate_math");
              auto divisorType = dyn_cast<RankedTensorType>(
                  i->getOperand(i->getNumOperands() - 1).getType());
              bool scalarDivisor =
                  code == "div" && (!divisorType || divisorType.getRank() == 1);
              double passes = fast ? (code == "sqrt" || scalarDivisor ? 1 : 2) : 6;
              mathPasses += passes;
              if (auto rate = intentdsl::quantity(ctx, "compute", "lane/ns")) {
                specialized += passes * n / *rate;
                ++count;
              }
              return;
            }
            std::string primitive;
            double invocationLanes = lanes, callsPerBatch = 1.;
            if(code=="reduce"){
              auto algebra=i->getAttrOfType<StringAttr>("algebra");
              if(algebra=="max_num")primitive="max";
              if(algebra=="sum")primitive="sum";
            } else if (code=="exp") primitive="exp";
            else if (code == "div") {
              auto out = dyn_cast<RankedTensorType>(i->getResult(0).getType());
              if (!out || out.getRank() != 2 || !out.getElementType().isF32())
                return;
              auto rank = [](Value v) -> int64_t {
                auto t = dyn_cast<RankedTensorType>(v.getType());
                return t ? t.getRank() : 0;
              };
              auto lhs = rank(i->getOperand(0)), rhs = rank(i->getOperand(1));
              primitive = lhs != 2   ? "div_rn_sv"
                          : rhs != 2 ? "div_rn_vs"
                                     : "div_rn_vv";
              if (lhs == 1 || rhs == 1) {
                invocationLanes = n;
                callsPerBatch = batch;
              }
            }
            if(primitive.empty())return;
            int low = invocationLanes <= 512 ? 64 : 512,
                high = invocationLanes <= 512 ? 512 : 2048;
            auto prefix="compute_"+primitive+"_";
            auto a=intentdsl::quantity(ctx,prefix+std::to_string(low),"ns");
            auto b=intentdsl::quantity(ctx,prefix+std::to_string(high),"ns");
            if(a && b){
              double u =
                  invocationLanes <= 512
                      ? std::clamp((invocationLanes - low) / (high - low), 0.,
                                   1.)
                      : (invocationLanes - low) / (high - low);
              specialized += callsPerBatch * ((*a) + ((*b) - (*a)) * u) / batch;
              ++count;
            }
          });
          e.terms["native_compute"]={specialized,"ns"};
          if (mathPasses)
            e.terms["vector_math_passes"] = {mathPasses, "count"};
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
  u.costModels.push_back(
      {{"cnnl-row-cost", "levels/storage/dsa::supply", "2",
        [](Operation *op, const StructuralCandidate &c,
           CostContext &ctx) -> llvm::Expected<CostEstimate> {
          auto *facts = ctx.target.services.get<intentdsl::CallFacts>();
          auto [m, n] = facts->geometry(op);
          auto fn = cast<func::FuncOp>(op);
          auto d = cast<DictionaryAttr>(c.value);
          auto kind = d.getAs<StringAttr>("library_kind").getValue();
          double bytes = 0;
          for (Value arg : fn.getArguments())
            if (isa<MemRefType>(arg.getType()))
              bytes += facts->viewBytes(arg);
          if (kind == "gather_rows")
            bytes = facts->viewBytes(fn.getArgument(1)) + 2 * facts->viewBytes(fn.getArgument(2));
          CostEstimate e;
          e.scope = "full_call";
          e.bottleneck = "external library";
          e.terms["bytes"] = {bytes, "B"};
          auto bw = intentdsl::quantity(ctx, "read_1", "B/ns"),
               launch = intentdsl::quantity(ctx, "launch", "ns"),
               startup = intentdsl::quantity(ctx, "first_io", "ns"),
               vec = intentdsl::quantity(ctx, "compute", "lane/ns");
          if (bw && launch && startup && vec) {
            double computeNs =
                (kind == "transpose" || kind == "gather_rows")
                    ? 0.0
                    : (3.0 * m * n) /
                          (std::max(1.0, double(facts->participants)) * (*vec));
            double timeNs =
                1.35 * (*launch) + bytes / (*bw) + computeNs;
            e.timeNs = timeNs;
            if (auto rows = d.getAs<ArrayAttr>("rows"); rows && !rows.empty()) {
              auto id = cast<DictionaryAttr>(rows[0]).getAs<IntegerAttr>("id");
              RowPrediction pred;
              pred.terms = e.terms;
              pred.timeNs = timeNs;
              e.rows[std::to_string(id.getInt())] = std::move(pred);
            }
          }
          return e;
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
