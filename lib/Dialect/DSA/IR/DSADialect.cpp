#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;
using namespace intent::dsa;
#include "Intent/Dialect/DSA/IR/DSADialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/DSA/IR/DSAAttrs.cpp.inc"
void IntentDSADialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Dialect/DSA/IR/DSAAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/Dialect/DSA/IR/DSAOps.cpp.inc"
      >();
}
LogicalResult EntryRequirementsAttr::verify(function_ref<InFlightDiagnostic()> error,
                                            bool disjointOutputs) {
  return disjointOutputs ? success() : error() << "DSA entry requires disjoint writable views";
}
LogicalResult ConfigurationAttr::verify(function_ref<InFlightDiagnostic()> error,
    int64_t tile, int64_t m, int64_t n, int64_t k, int64_t region, int64_t tasks, int64_t bytes) {
  if (tile <= 0 || tile % 64 || m <= 0 || n <= 0 || k <= 0 || region <= 0 || tasks <= 0 || bytes <= 0)
    return error() << "DSA configuration requires positive blocks/resources and a vector tile divisible by 64";
  return success();
}
