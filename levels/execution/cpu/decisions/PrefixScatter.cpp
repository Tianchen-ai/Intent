#include "compilerinfra/Assembly.h"
namespace CI_UNIT_NAMESPACE {
void registerPrefix(compilerinfra::Unit &,bool);
void registerPrefixScatter(compilerinfra::Unit &u){registerPrefix(u,true);}
}
