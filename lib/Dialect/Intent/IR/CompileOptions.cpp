#include "Intent/Dialect/Intent/IR/CompileOptions.h"

using namespace mlir;

namespace intent {
FailureOr<CompileOptionsAttr> readCompileOptions(Operation *operation) {
  Operation *owner = operation;
  while (owner && !isa<ModuleOp>(owner)) owner = owner->getParentOp();
  auto options = owner ? owner->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr) : CompileOptionsAttr{};
  if (!options) return operation->emitOpError("requires bound intent.compile_options on its module"), failure();
  return options;
}

llvm::json::Object serializeCompileOptions(CompileOptionsAttr options) {
  return llvm::json::Object{{"numerics", stringifyNumericsMode(options.getNumerics())},
          {"online_reduction", options.getOnlineReduction()},
          {"optimization_remarks", options.getOptimizationRemarks()}};
}
}
