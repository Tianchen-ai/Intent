#include "Intent/Dialect/Intent/IR/CompileOptions.h"

using namespace mlir;

namespace intent {
LogicalResult CompileOptionsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
    NumericsMode numerics, DictionaryAttr optimizations, bool optimizationRemarks) {
  if (!optimizations) return emitError() << "optimization controls require a dictionary";
  for (NamedAttribute entry : optimizations)
    if (entry.getName().empty() || !mlir::isa<BoolAttr>(entry.getValue()))
      return emitError() << "optimization controls require named boolean entries";
  return success();
}

FailureOr<CompileOptionsAttr> readCompileOptions(Operation *operation) {
  Operation *owner = operation;
  while (owner && !isa<ModuleOp>(owner)) owner = owner->getParentOp();
  auto options = owner ? owner->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr) : CompileOptionsAttr{};
  if (!options) return operation->emitOpError("requires bound intent.compile_options on its module"), failure();
  return options;
}

llvm::json::Object serializeCompileOptions(CompileOptionsAttr options) {
  llvm::json::Object controls;
  for (NamedAttribute entry : options.getOptimizations())
    controls[entry.getName().strref()] = cast<BoolAttr>(entry.getValue()).getValue();
  return llvm::json::Object{{"numerics", stringifyNumericsMode(options.getNumerics())},
          {"optimizations", std::move(controls)},
          {"optimization_remarks", options.getOptimizationRemarks()}};
}
}
