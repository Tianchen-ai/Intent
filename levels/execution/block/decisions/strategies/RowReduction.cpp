#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerRowStrategy(Unit &unit, StringRef mode, bool tuple) {
  Strategy strategy;
  strategy.name = mode.str();
  strategy.family = unit.qualify("structure");
  auto propose = [tuple](Operation *op, const TargetProfile &target,
                        StringRef mode) -> llvm::Expected<Attribute> {
    bool reduces = false, hasTuple=false;
    op->walk([&](Operation *v) {
      auto n = v->getName().getStringRef();
      reduces |= n == "block.sum" || n == "block.max_num";
      hasTuple |= n == "block.welford";
    });
    if (tuple != hasTuple || (!reduces && !hasTuple))
      return Attribute{};
    auto *caps = target.services.get<intentdsl::RowCapacity>();
    if (!caps || caps->rows.empty())
      return llvm::createStringError(
          "row reduction requires a bound GPU capability table");
    Builder b(op->getContext());
    auto extent=caps->width(op);
    if(!extent)return extent.takeError();
    unsigned width=0, limit=0, previous=0;
    for(auto &r:caps->rows) {
      limit=std::max(limit,r.width);
      if(r.width>=*extent && (!width || r.width<width)) width=r.width;
    }
    bool mustStream=false;
    if(caps->stagedInputBudget && width) {
      llvm::SmallDenseSet<Value> slots;
      op->walk([&](Operation *read){
        if(read->getName().getStringRef()=="block.read")slots.insert(read->getOperand(0));
      });
      uint64_t minimum=0;
      for(auto slot:slots)
        minimum+=llvm::divideCeil(cast<MemRefType>(slot.getType()).getElementTypeBitWidth(),8u);
      mustStream=minimum>caps->stagedInputBudget/width;
    }
    bool resident = width && !mustStream && mode!="stream-row";
    if(!resident && mode=="resident-row") return Attribute{};
    if(!resident) width=limit;
    for(auto &r:caps->rows) if(r.width<width)previous=std::max(previous,r.width);
    int64_t lo=resident?(previous?previous+1:0):(mode=="stream-row"?0:limit+1);
    auto guard=b.getDenseI64ArrayAttr({lo,resident?int64_t(width):-1});
    SmallVector<Attribute> rows;
    for (const auto &r : caps->rows) {
      if (r.width != width && !(mustStream && llvm::is_contained(caps->streamingWidths,r.width)))
        continue;
      rows.push_back(b.getDictionaryAttr(
          {b.getNamedAttr("id", b.getI32IntegerAttr(r.id)),
           b.getNamedAttr("rows", b.getI32IntegerAttr(r.rows)),
           b.getNamedAttr("width", b.getI32IntegerAttr(r.width)),
           b.getNamedAttr("payload", b.getStringAttr(r.payload))}));
    }
    if (rows.empty())
      return llvm::createStringError("no legal row configurations");
    return Attribute(b.getDictionaryAttr(
        {b.getNamedAttr("guard",guard),b.getNamedAttr("topology",
                        b.getStringAttr(resident ? "resident" : "stream")),
         b.getNamedAttr("rows", b.getArrayAttr(rows))}));
  };
  strategy.enumerate=[propose,mode](Operation *op,const TargetProfile &p,AnalysisManager &)->llvm::Expected<std::vector<StructuralCandidate>>{
    std::vector<StructuralCandidate> candidates;
    for(StringRef m:{"resident-row","stream-row"}){
      if((mode=="resident-row" || mode=="stream-row") && mode!=m)continue;
      auto v=propose(op,p,m);if(!v)return v.takeError();if(!*v)continue;
      auto d=cast<DictionaryAttr>(*v);auto topology=d.getAs<StringAttr>("topology").str();
      std::map<int64_t,SmallVector<Attribute>> groups;
      for(auto r:d.getAs<ArrayAttr>("rows"))groups[cast<DictionaryAttr>(r).getAs<IntegerAttr>("rows").getInt()].push_back(r);
      for(auto &[n,rs]:groups){NamedAttrList attrs(d);attrs.set("rows",ArrayAttr::get(op->getContext(),rs));
        candidates.push_back({topology+"-"+std::to_string(n),topology,attrs.getDictionary(op->getContext()),{"legal-width-group"}});}
    }
    return candidates;};
  unit.strategies.push_back({std::move(strategy), COMPILERINFRA_SOURCE});
}
void registerRowReduction(Unit &unit) {
  for (StringRef mode : {"row-reduction", "resident-row", "stream-row"})
    registerRowStrategy(unit,mode,false);
}
} // namespace CI_UNIT_NAMESPACE
