#include "../Resident.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
namespace CI_UNIT_NAMESPACE {
void registerKeyReduction(compilerinfra::Unit &unit) {
  registerResident(unit,"counts-by-key",[](mlir::Operation *op){
    return contains(op,{"count_keys"});
  },true);
  registerResident(unit,"complete-bins",[](mlir::Operation *op){return contains(op,{"column_sum"});},true);
  using namespace mlir;
  using namespace compilerinfra;
  Strategy s;s.name="column-panels";s.family=unit.qualify("structure");
  s.enumerate=[](Operation *op,const TargetProfile &p,AnalysisManager &)->llvm::Expected<std::vector<StructuralCandidate>>{
    std::vector<StructuralCandidate> cs;
    auto caps=p.services.get<intentdsl::RowCapacity>();
    if(!contains(op,{"column_sum"}) || !caps || caps->independentWidths.empty())return cs;
    Builder b(op->getContext());
    for(auto width:caps->independentWidths){
      SmallVector<Attribute> rows;
      for(auto &r:caps->rows)if(r.width==width && r.rows==1)
        rows.push_back(b.getDictionaryAttr({b.getNamedAttr("id",b.getI32IntegerAttr(r.id)),b.getNamedAttr("width",b.getI32IntegerAttr(r.width)),b.getNamedAttr("rows",b.getI32IntegerAttr(1)),b.getNamedAttr("payload",b.getStringAttr(r.payload))}));
      if(rows.empty())continue;
      auto d=b.getDictionaryAttr({b.getNamedAttr("rows",b.getArrayAttr(rows)),b.getNamedAttr("walk",b.getStringAttr("column-panels")),
          b.getNamedAttr("topology",b.getStringAttr("stream")),b.getNamedAttr("guard",b.getDenseI64ArrayAttr({0,-1}))});
      cs.push_back({"column-panels-"+std::to_string(width),"column-panels",d,{"independent-bins"}});
    }
    return cs;
  };
  unit.strategies.push_back({std::move(s),COMPILERINFRA_SOURCE});
}
}
