#include "compilerinfra/Assembly.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
llvm::Expected<Attribute> proposeRows(Operation *,const TargetProfile &,bool);
void registerRowReduction(Unit &unit){
  unit.strategies.push_back({{"row-reduction",unit.qualify("partition"),[](Operation *op,const TargetProfile &p,AnalysisManager &){
    return proposeRows(op,p,true);}},COMPILERINFRA_SOURCE});
}
}
