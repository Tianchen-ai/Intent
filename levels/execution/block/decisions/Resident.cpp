#include "Resident.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerResident(Unit &u,StringRef name,RowMatch accepts,bool singleOwner) {
  Strategy strategy;
  strategy.name=name.str();strategy.family=u.qualify("structure");
  strategy.enumerate=[accepts,singleOwner](Operation *op,const TargetProfile &p,AnalysisManager &)->llvm::Expected<std::vector<StructuralCandidate>>{
    std::vector<StructuralCandidate> choices;
    if(!accepts(op))return choices;
    auto *caps=p.services.get<intentdsl::RowCapacity>();
    if(!caps)return llvm::createStringError("resident collective requires target row capacities");
    auto n=caps->width(op);if(!n)return n.takeError();
    unsigned width=0,last=0;
    for(auto &r:caps->rows)if(r.width>=*n && (!width || width>r.width))width=r.width;
    if(!width)return llvm::createStringError("whole-row collective exceeds the available native tile widths");
    for(auto &r:caps->rows)if(r.width<width)last=std::max(last,r.width);
    Builder b(op->getContext());
    for(auto &r:caps->rows)if(r.width==width && (!singleOwner || r.rows==1) && caps->acceptsGroup(op,r.rows)){
      auto row=b.getDictionaryAttr({b.getNamedAttr("rows",b.getI32IntegerAttr(r.rows)),b.getNamedAttr("id",b.getI32IntegerAttr(r.id)),
        b.getNamedAttr("width",b.getI32IntegerAttr(width)),b.getNamedAttr("payload",b.getStringAttr(r.payload))});
      auto d=b.getDictionaryAttr({b.getNamedAttr("topology",b.getStringAttr("resident")),b.getNamedAttr("rows",b.getArrayAttr({row})),
        b.getNamedAttr("guard",b.getDenseI64ArrayAttr({last?last+1:0,width}))});
      choices.push_back({"resident-"+std::to_string(r.rows),"resident",d,{"legal-width-group"}});
    }
    return choices;
  };
  u.strategies.push_back({std::move(strategy),COMPILERINFRA_SOURCE});
}
}
