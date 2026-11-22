#include "compilerinfra/Assembly.h"
namespace CI_UNIT_NAMESPACE {
void registerPointwise(compilerinfra::Unit &,llvm::StringRef);
void registerIndirect(compilerinfra::Unit &u){registerPointwise(u,"indirect");}
}
