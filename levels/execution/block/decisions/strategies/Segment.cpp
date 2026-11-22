#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerSegment(Unit &unit) {
  Strategy s;
  s.name="segment-members"; s.family=unit.qualify("structure");
  s.enumerate=[](Operation *op,const TargetProfile &profile,AnalysisManager &)
      ->llvm::Expected<std::vector<StructuralCandidate>> {
    std::vector<StructuralCandidate> candidates;
    if (!op->walk([](Operation *v){return v->getName().stripDialect()=="ordered_sum" ?
            WalkResult::interrupt():WalkResult::advance();}).wasInterrupted()) return candidates;
    auto *caps=profile.services.get<intentdsl::RowCapacity>();
    if(!caps)return llvm::createStringError("segment mapping requires target tile capacities");
    bool grid=!caps->independentWidths.empty();
    auto extent=caps->width(op);if(!extent)return extent.takeError();
    Builder b(op->getContext());SmallVector<Attribute> rows;
    unsigned width=0;
    if(!grid)for(auto &r:caps->rows)if(r.width>=*extent && (!width || width>r.width))width=r.width;
    int64_t previous=0;
    if(width) for(auto &r:caps->rows)if(r.width<width)previous=std::max(previous,int64_t(r.width));
    bool streamed=!grid && !width;
    for(auto &r:caps->rows) {
      if(!caps->acceptsGroup(op,r.rows) || (cast<MemRefType>(op->getOperand(0).getType()).getRank()==1 && r.rows!=1))continue;
      if(grid ? !llvm::is_contained(caps->independentWidths,r.width) :
           streamed ? !llvm::is_contained(caps->streamingWidths,r.width) : width!=r.width) continue;
      rows.push_back(b.getDictionaryAttr({b.getNamedAttr("id",b.getI32IntegerAttr(r.id)),b.getNamedAttr("rows",b.getI32IntegerAttr(r.rows)),
          b.getNamedAttr("width",b.getI32IntegerAttr(r.width)),b.getNamedAttr("payload",b.getStringAttr(r.payload))}));
    }
    if(rows.empty())return candidates;
    StringRef walk=grid?"segment-tile":streamed?"segment-stream":"segment-row";
    auto d=b.getDictionaryAttr({b.getNamedAttr("rows",b.getArrayAttr(rows)),b.getNamedAttr("walk",b.getStringAttr(walk)),
        b.getNamedAttr("topology",b.getStringAttr(width?"resident":"stream")),b.getNamedAttr("guard",b.getDenseI64ArrayAttr({previous?previous+1:0,width?int64_t(width):-1}))});
    candidates.push_back({walk.str(),walk.str(),d,{"distinct-segment-outputs"}});
    return candidates;
  };
  unit.strategies.push_back({std::move(s),COMPILERINFRA_SOURCE});
}
}
