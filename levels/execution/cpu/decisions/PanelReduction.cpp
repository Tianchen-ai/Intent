#include "compilerinfra/Assembly.h"
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
llvm::Expected<std::vector<StructuralCandidate>>
enumerateReductionRows(Operation *, const TargetProfile &, bool,
                       llvm::StringRef);
void registerPanelReduction(Unit &unit) {
  Strategy s;
  s.name = "panel-reduction";
  s.family = unit.qualify("partition");
  std::string sid = unit.qualify("panel-reduction");
  s.enumerate = [sid](Operation *op, const TargetProfile &p,
                      AnalysisManager &) {
    return enumerateReductionRows(op, p, false, sid);
  };
  unit.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
}
}
