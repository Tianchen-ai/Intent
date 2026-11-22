#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerPrefix(Unit &u,bool scatter){
  Strategy s;
  s.name=scatter?"scatter-reduction":"row-scan";s.family=u.qualify("partition");
  s.propose=[scatter](Operation *op,const TargetProfile &p,AnalysisManager &)->llvm::Expected<Attribute>{
    if(scatter!=op->walk([](Operation *v){return v->getName().stripDialect()=="prefix_scatter"?WalkResult::interrupt():WalkResult::advance();}).wasInterrupted())return Attribute{};
    if(!op->walk([](Operation *v){return v->getName().getStringRef()=="cpu.scan_sum"?WalkResult::interrupt():WalkResult::advance();}).wasInterrupted())return Attribute{};
    auto *m=p.services.get<intentdsl::cpu::RowMachine>();
    if(!m)return llvm::createStringError("missing CPU prefix capacity");
    Builder b(op->getContext());
    return Attribute(intentdsl::cpu::partition(b,"source",m->configurations(op->getContext(),false)));
  };
  u.strategies.push_back({std::move(s),COMPILERINFRA_SOURCE});
}
void registerScan(Unit &u){registerPrefix(u,false);}
}
