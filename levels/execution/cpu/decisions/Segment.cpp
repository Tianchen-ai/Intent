#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void registerSegment(Unit &u){
  Strategy s;
  s.name="segment-reduction";s.family=u.qualify("partition");
  s.propose=[](Operation *op,const TargetProfile &p,AnalysisManager &)->llvm::Expected<Attribute>{
    if(!op->walk([](Operation *v){return v->getName().stripDialect()=="ordered_sum"?WalkResult::interrupt():WalkResult::advance();}).wasInterrupted())return Attribute{};
    auto *machine=p.services.get<intentdsl::cpu::RowMachine>();
    if(!machine)return llvm::createStringError("missing task capacity for segment outputs");
    Builder b(op->getContext());
    NamedAttrList layout(intentdsl::cpu::partition(b,"source",machine->configurations(op->getContext(),false)));
    if(cast<MemRefType>(op->getOperand(0).getType()).getRank()==1)layout.set("member_tasks",b.getUnitAttr());
    return Attribute(layout.getDictionary(op->getContext()));
  };
  u.strategies.push_back({std::move(s),COMPILERINFRA_SOURCE});
}
}
