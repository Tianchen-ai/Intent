#include "intentdsl/levels/algorithm/kir/ir/Permutation.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "mlir/IR/Builders.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void registerElementTraversal(Unit &u,bool indirect) {
  Strategy s;
  s.name = indirect?"indexed-rows":"pointwise";
  s.family = u.qualify("structure");
  auto original = [indirect](Operation *op, const TargetProfile &p,
                 AnalysisManager &) -> llvm::Expected<Attribute> {
    bool writes = false, collective = false, gathers = false, effects = false;
    op->walk([&](Operation *o) {
      auto n = o->getName().stripDialect();
      writes |= n == "write";
      gathers |= n == "gather_row";
      effects |= n == "atomic_add" || n == "unique_put" ||
                 n == "prefix_scatter" || n == "fill_ranges" ||
                 n == "write_offsets";
      collective |= n == "sum" || n == "max_num" || n == "welford" || n=="scan_sum" || n=="ordered_sum" || n=="count_keys" || n=="column_sum";
    });
    if (!writes || collective || effects || gathers!=indirect)
      return Attribute{};
    auto *capacity = p.services.get<intentdsl::RowCapacity>();
    if (!capacity || capacity->rows.empty())
      return llvm::createStringError(
          "pointwise mapping needs registered tile rows");
    // Independent lanes require no cross-tile carry. Tile the inner extent,
    // including short tails, and use the same traversal for all widths.
    auto extent=capacity->width(op);
    if(!extent)return extent.takeError();
    unsigned limit = 0, width = 0, previous = 0;
    for (auto &r : capacity->rows) {
      limit = std::max(limit, r.width);
      if (r.width >= *extent && (!width || r.width < width))
        width = r.width;
    }
    bool fits = bool(width);
    if (!fits)
      width = limit;
    for (auto &r : capacity->rows)
      if (r.width < width)
        previous = std::max(previous, r.width);
    Builder b(op->getContext());
    SmallVector<Attribute> rows;
    for (auto &r : capacity->rows)
      if ((r.width == width ||
           (!fits && llvm::is_contained(capacity->streamingWidths, r.width))) &&
          (cast<MemRefType>(op->getOperand(0).getType()).getRank() > 1 ||
           r.rows == 1) && capacity->acceptsGroup(op, r.rows))
        rows.push_back(b.getDictionaryAttr(
            {b.getNamedAttr("id", b.getI32IntegerAttr(r.id)),
             b.getNamedAttr("rows", b.getI32IntegerAttr(r.rows)),
             b.getNamedAttr("width", b.getI32IntegerAttr(r.width)),
             b.getNamedAttr("payload", b.getStringAttr(r.payload))}));
    return Attribute(b.getDictionaryAttr(
        {b.getNamedAttr("guard",
                        b.getDenseI64ArrayAttr(
                            {fits ? (previous ? int64_t(previous) + 1 : 0)
                                  : int64_t(limit) + 1,
                             fits ? int64_t(width) : -1})),
         b.getNamedAttr("topology",
                        b.getStringAttr(fits ? "resident" : "stream")),
         b.getNamedAttr("rows", b.getArrayAttr(rows))}));
  };
  s.enumerate=[original,indirect](Operation *op,const TargetProfile &p,AnalysisManager &a)->llvm::Expected<std::vector<StructuralCandidate>>{
    std::vector<StructuralCandidate> cs;auto v=original(op,p,a);if(!v)return v.takeError();
    if(!*v)return cs;
    Builder b(op->getContext());auto caps=p.services.get<intentdsl::RowCapacity>();
    for(StringRef walk:{"source","destination","tile","grid-tile"}){
      if(walk=="grid-tile" && caps->independentWidths.empty()) continue;
      if(walk!="source" && !intentdsl::rectangularCopy(op))continue;
      NamedAttrList d(cast<DictionaryAttr>(*v));d.set("walk",b.getStringAttr(walk));
      if(walk!="source"){
        SmallVector<Attribute> rows;
        for(auto &r:caps->rows)if((walk=="grid-tile"?llvm::is_contained(caps->independentWidths,r.width):r.width==32) && (walk=="destination"?r.rows==1:r.rows>1))
          rows.push_back(b.getDictionaryAttr({b.getNamedAttr("width",b.getI32IntegerAttr(r.width)),b.getNamedAttr("rows",b.getI32IntegerAttr(r.rows)),b.getNamedAttr("id",b.getI32IntegerAttr(r.id)),b.getNamedAttr("payload",b.getStringAttr(r.payload))}));
        if(rows.empty())continue;
        d.set("guard",b.getDenseI64ArrayAttr({0,-1}));d.set("rows",b.getArrayAttr(rows));d.set("topology",b.getStringAttr("stream"));
      }
      cs.push_back({walk.str(),walk.str(),d.getDictionary(op->getContext()),{walk=="source"?"independent-members":"bijective-rectangular-copy"}});
    }
    auto fn = op->getParentOfType<func::FuncOp>();
    bool rank1 = cast<MemRefType>(op->getOperand(0).getType()).getRank() == 1;
    if (!indirect && !intentdsl::rectangularCopy(op) && rank1 &&
        !caps->independentWidths.empty() && !fn->hasAttr("kir.iteration") &&
        !fn->hasAttr("kir.owner_axis")) {
      for (unsigned width : {128u, 256u, 512u, 1024u}) {
        SmallVector<Attribute> rows;
        for (auto &r : caps->rows)
          if (r.width == width && r.rows == 1)
            rows.push_back(b.getDictionaryAttr(
                {b.getNamedAttr("id", b.getI32IntegerAttr(r.id)),
                 b.getNamedAttr("rows", b.getI32IntegerAttr(1)),
                 b.getNamedAttr("width", b.getI32IntegerAttr(r.width)),
                 b.getNamedAttr("payload", b.getStringAttr(r.payload))}));
        if (rows.empty())
          continue;
        auto d = b.getDictionaryAttr(
            {b.getNamedAttr("guard", b.getDenseI64ArrayAttr({0, -1})),
             b.getNamedAttr("topology", b.getStringAttr("stream")),
             b.getNamedAttr("walk", b.getStringAttr("column-panels")),
             b.getNamedAttr("rows", b.getArrayAttr(rows))});
        cs.push_back({"column-panels-" + std::to_string(width),
                      "column-panels",
                      d,
                      {"independent-members"}});
      }
    }
    return cs;};
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
}
void registerPointwise(Unit &u){registerElementTraversal(u,false);}
} // namespace CI_UNIT_NAMESPACE
