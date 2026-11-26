#include "../Resident.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/decisions/KeyPanels.h"
namespace CI_UNIT_NAMESPACE {
void registerKeyReduction(compilerinfra::Unit &unit) {
  using namespace mlir;
  using namespace compilerinfra;
  Strategy bins;
  bins.name="counts-by-key";bins.family=unit.qualify("structure");
  bins.enumerate=[](Operation *op,const TargetProfile &p,AnalysisManager &)->llvm::Expected<std::vector<StructuralCandidate>>{
    std::vector<StructuralCandidate> choices;
    if(!contains(op,{"count_keys"}))return choices;
    auto rows=residentCandidates(op,p,true);
    if(!rows)return rows.takeError();
    auto caps=p.services.get<intentdsl::RowCapacity>();
    bool direct=caps->globalAtomicBins;
    op->walk([&](Operation *k){if(k->getName().stripDialect()=="count_keys")
      direct &= k->hasOneUse() && (*k->getUsers().begin())->getName().stripDialect()=="write";});
    for(auto &row:*rows)for(StringRef mode:{"private-merge","global-atomic"}){
      if(mode=="global-atomic" && !direct)continue;
      NamedAttrList d(cast<DictionaryAttr>(row.value));Builder b(op->getContext());
      auto params=cast<DictionaryAttr>(cast<ArrayAttr>(d.get("rows"))[0]);
      if(mode=="private-merge" && caps->nativePrivateHistogram && caps->privateBinCapacity &&
         4*params.getAs<IntegerAttr>("width").getInt()>*caps->privateBinCapacity)continue;
      d.set("bins",b.getStringAttr(mode));
      choices.push_back({mode.str(),mode.str(),d.getDictionary(op->getContext()),{"exclusive-bin-row"}});
    }
    return choices;
  };
  unit.strategies.push_back({std::move(bins),COMPILERINFRA_SOURCE});
  registerResident(unit,"complete-bins",[](mlir::Operation *op){return contains(op,{"column_sum"});},true);
  using namespace mlir;
  using namespace compilerinfra;
  Strategy s;s.name="column-panels";s.family=unit.qualify("structure");
  s.enumerate = [](Operation *op, const TargetProfile &p, AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    std::vector<StructuralCandidate> cs;
    auto caps = p.services.get<intentdsl::RowCapacity>();
    bool keyed = contains(op, {"count_keys"});
    if (!caps || caps->independentWidths.empty() ||
        (!contains(op, {"column_sum"}) && !keyed))
      return cs;
    Builder b(op->getContext());
    unsigned elemBytes = 0;
    op->walk([&](Operation *k) {
      if (k->getName().stripDialect() == "count_keys") {
        elemBytes = sizeof(int32_t);
      } else if (k->getName().stripDialect() == "column_sum") {
        auto ty = cast<MemRefType>(k->getOperand(0).getType());
        elemBytes = std::max<unsigned>(
            elemBytes, llvm::divideCeil(ty.getElementTypeBitWidth(), 8));
      }
    });
    for (unsigned width : {2u, 4u, 8u, 16u, 32u, 64u, 128u}) {
      if (elemBytes && width * elemBytes < caps->storeSectorBytes)
        continue;
      SmallVector<Attribute> rows;
      for (auto &r : caps->rows)
        if (r.width == width && r.rows == 1)
          rows.push_back(b.getDictionaryAttr(
              {b.getNamedAttr("id", b.getI32IntegerAttr(r.id)),
               b.getNamedAttr("width", b.getI32IntegerAttr(r.width)),
               b.getNamedAttr("rows", b.getI32IntegerAttr(1)),
               b.getNamedAttr("payload", b.getStringAttr(r.payload))}));
      if (rows.empty())
        continue;
      auto d = b.getDictionaryAttr(
          {b.getNamedAttr("rows", b.getArrayAttr(rows)),
           b.getNamedAttr("walk", b.getStringAttr("column-panels")),
           b.getNamedAttr("topology", b.getStringAttr("stream")),
           b.getNamedAttr("guard", b.getDenseI64ArrayAttr({0, -1}))});
      if (keyed) {
        int64_t lanes = intentdsl::block::privateKeyPanel(op, p, width);
        if (!lanes)
          continue;
        NamedAttrList params(d);
        params.set("bins", b.getStringAttr("private-merge"));
        params.set("key_panel", b.getI64IntegerAttr(lanes));
        d = params.getDictionary(op->getContext());
      }
      cs.push_back({"column-panels-" + std::to_string(width),
                    "column-panels",
                    d,
                    {"independent-bins"}});
    }
    return cs;
  };
  unit.strategies.push_back({std::move(s),COMPILERINFRA_SOURCE});
}
}
