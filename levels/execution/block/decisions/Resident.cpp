#include "Resident.h"
#include "intentdsl/levels/execution/block/decisions/KeyPanels.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include <map>

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
llvm::Expected<std::vector<StructuralCandidate>> residentCandidates(Operation *op,const TargetProfile &p,bool singleOwner){
    std::vector<StructuralCandidate> choices;
    auto *caps=p.services.get<intentdsl::RowCapacity>();
    if(!caps)return llvm::createStringError("resident collective requires target row capacities");
    auto n=caps->width(op);if(!n)return n.takeError();
    unsigned width=0,last=0;
    for(auto &r:caps->rows)if(r.width>=*n && (!width || width>r.width))width=r.width;
    if (!width) {
      // These reductions may still cover the row with independent bin tiles.
      if (!caps->independentWidths.empty() &&
          (contains(op, {"column_sum"}) || intentdsl::block::privateKeyPanel(op, p, 1)))
        return choices;
      return llvm::createStringError("whole-row collective exceeds the available native tile widths");
    }
    for(auto &r:caps->rows)if(r.width<width)last=std::max(last,r.width);
    Builder b(op->getContext());
    std::map<unsigned, SmallVector<Attribute>> groups;
    for(auto &r:caps->rows)if(r.width==width && (!singleOwner || r.rows==1) && caps->acceptsGroup(op,r.rows)){
      groups[r.rows].push_back(b.getDictionaryAttr({b.getNamedAttr("rows",b.getI32IntegerAttr(r.rows)),b.getNamedAttr("id",b.getI32IntegerAttr(r.id)),
        b.getNamedAttr("width",b.getI32IntegerAttr(width)),b.getNamedAttr("payload",b.getStringAttr(r.payload))}));
    }
    for (auto &[rowCount, rs] : groups) {
      auto d=b.getDictionaryAttr({b.getNamedAttr("topology",b.getStringAttr("resident")),b.getNamedAttr("rows",b.getArrayAttr(rs)),
        b.getNamedAttr("guard",b.getDenseI64ArrayAttr({last?last+1:0,width}))});
      choices.push_back({"resident-"+std::to_string(rowCount),"resident",d,{"legal-width-group"}});
    }
    return choices;
}
void registerResident(Unit &u,StringRef name,RowMatch accepts,bool singleOwner) {
  Strategy strategy;
  strategy.name=name.str();strategy.family=u.qualify("structure");
  strategy.enumerate=[accepts,singleOwner](Operation *op,const TargetProfile &p,AnalysisManager &)->llvm::Expected<std::vector<StructuralCandidate>>{
    if(!accepts(op))return std::vector<StructuralCandidate>{};
    return residentCandidates(op,p,singleOwner);
  };
  u.strategies.push_back({std::move(strategy),COMPILERINFRA_SOURCE});
}
}
