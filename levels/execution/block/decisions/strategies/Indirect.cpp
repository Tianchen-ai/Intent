#include "compilerinfra/Assembly.h"
namespace CI_UNIT_NAMESPACE {
void registerElementTraversal(compilerinfra::Unit &,bool);
void registerIndirect(compilerinfra::Unit &u){registerElementTraversal(u,true);}
}
