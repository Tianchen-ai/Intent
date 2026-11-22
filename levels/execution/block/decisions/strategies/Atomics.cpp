#include "../Resident.h"
namespace CI_UNIT_NAMESPACE {
void registerAtomics(compilerinfra::Unit &unit) {
  registerResident(unit,"keyed-atomics",[](mlir::Operation *op){
    return contains(op,{"atomic_add","fill_ranges"});
  });
}
}
