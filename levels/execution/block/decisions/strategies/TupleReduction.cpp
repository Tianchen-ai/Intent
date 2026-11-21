#include "compilerinfra/Assembly.h"
namespace CI_UNIT_NAMESPACE {
void registerRowStrategy(compilerinfra::Unit &, llvm::StringRef, bool);
void registerTupleReduction(compilerinfra::Unit &unit) {
  registerRowStrategy(unit,"tuple-reduction",true);
}
}
