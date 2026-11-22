#include "../Resident.h"
namespace CI_UNIT_NAMESPACE {
void registerScan(compilerinfra::Unit &unit) {
  registerResident(unit,"row-scan",[](mlir::Operation *op){
    return contains(op,{"scan_sum"}) && !contains(op,{"prefix_scatter"});
  });
}
}
