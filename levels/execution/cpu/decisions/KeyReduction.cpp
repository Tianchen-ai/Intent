#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerKeyReduction(Unit &u){
  Strategy s;
  s.name="key-reduction";s.family=u.qualify("partition");
  s.propose=[](Operation *op,const TargetProfile &p,AnalysisManager &)->llvm::Expected<Attribute>{
    auto found=op->walk([](Operation *v){auto n=v->getName().stripDialect();return n=="count_keys" || n=="column_sum"?WalkResult::interrupt():WalkResult::advance();});
    if(!found.wasInterrupted())return Attribute{};
    auto *m=p.services.get<intentdsl::cpu::RowMachine>();if(!m)return llvm::createStringError("missing key counting task resources");
    Builder b(op->getContext());
    return Attribute(intentdsl::cpu::partition(b,"source",m->configurations(op->getContext(),false)));
  };
  u.strategies.push_back({std::move(s),COMPILERINFRA_SOURCE});
}
}
