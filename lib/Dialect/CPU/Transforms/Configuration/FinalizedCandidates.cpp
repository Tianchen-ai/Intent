#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "Intent/Dialect/CPU/Transforms/Configuration/FinalizedCandidates.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::cpu {
namespace {

DictionaryAttr executionAttributes(func::FuncOp function) {
  NamedAttrList attributes(function->getAttrs());
  attributes.erase(SymbolTable::getSymbolAttrName());
  // These bindings describe the realized candidate for artifact inspection.
  // At this boundary they no longer select operations, loops or resources.
  attributes.erase("intent_cpu.configuration");
  attributes.erase("intent_cpu.implementations");
  return attributes.getDictionary(function.getContext());
}

bool equivalent(func::FuncOp lhs, func::FuncOp rhs) {
  return lhs.getFunctionType() == rhs.getFunctionType() &&
      executionAttributes(lhs) == executionAttributes(rhs) &&
      OperationEquivalence::isRegionEquivalentTo(
          &lhs.getBody(), &rhs.getBody(), OperationEquivalence::IgnoreLocations);
}

} // namespace

void deduplicateFinalizedCandidates(ModuleOp module) {
  SmallVector<func::FuncOp> retained;
  for (func::FuncOp function : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
    if (function.isExternal() ||
        !function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration") ||
        !function->getAttrOfType<ArrayAttr>("intent_cpu.implementations") ||
        !getPublicInterface(function))
      continue;
    // Do not infer a correspondence between separate task/callee graphs, or
    // change an entry whose symbol identity is observed elsewhere in the IR.
    if (!SymbolTable::symbolKnownUseEmpty(function, module)) continue;
    if (llvm::any_of(retained, [&](func::FuncOp previous) { return equivalent(previous, function); }))
      function.erase();
    else
      retained.push_back(function);
  }
}

} // namespace intent::cpu
