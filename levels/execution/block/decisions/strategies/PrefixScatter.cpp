#include "../Resident.h"
namespace CI_UNIT_NAMESPACE {
void registerPrefixScatter(compilerinfra::Unit &unit) {
  registerResident(unit,"prefix-scatter",[](mlir::Operation *op){
    return contains(op,{"prefix_scatter"});
  });
}
}
