#include "Intent/Target/Weft/IR/WeftDialect.h"
#include "Intent/Target/Weft/IR/WeftAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace intent::weft_provider;

#include "Intent/Target/Weft/IR/WeftDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Intent/Target/Weft/IR/WeftAttrs.cpp.inc"

void IntentWeftDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Target/Weft/IR/WeftAttrs.cpp.inc"
  >();
}

LogicalResult TaskBindingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, SymbolRefAttr hostEntry,
    SymbolRefAttr hostCallee, SymbolRefAttr deviceKernel, int64_t ordinal,
    int64_t coordinateArgument) {
  if (!hostEntry || !hostCallee || !deviceKernel || ordinal < 0 ||
      (coordinateArgument != -1 && coordinateArgument != 0))
    return emitError() << "task binding requires host/device symbols, an ordinal and an optional coordinate slot";
  return success();
}

LogicalResult ArgumentAlignmentAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, int64_t bytes) {
  if (bytes <= 0) return emitError() << "public argument alignment must be positive";
  return success();
}
