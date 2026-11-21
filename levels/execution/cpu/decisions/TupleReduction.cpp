#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void registerTupleReduction(Unit &unit) {
  Strategy s;
  s.name="tuple-reduction";
  s.family=unit.qualify("partition");
  s.propose=[](Operation *op,const TargetProfile &profile,AnalysisManager &)->llvm::Expected<Attribute> {
    bool matches=false;
    op->walk([&](Operation *v){ matches |= v->getName().getStringRef()=="cpu.welford"; });
    if(!matches)return Attribute{};
    auto *caps=profile.services.get<intentdsl::cpu::RowMachine>();
    if(!caps)return llvm::createStringError("CPU tuple reduction requires row capacity");
    Builder b(op->getContext());
    return Attribute(intentdsl::cpu::partition(b,"source",caps->configurations(op->getContext(),false)));
  };
  unit.strategies.push_back({std::move(s),COMPILERINFRA_SOURCE});
}
}
