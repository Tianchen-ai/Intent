#include "Construction.h"
#include "Intent/Conversion/ScalarLowering.h"

using namespace mlir;
namespace intent {

LogicalResult lowerCanonicalKIRToDSA(ModuleOp module, dsa::ConfigurationAttr configuration,
                                   DictionaryAttr shapes, DictionaryAttr strides) {
  CanonicalKernelAnalysis analysis(module);
  if (failed(analysis.verify())) return failure();
  OwningOpRef<ModuleOp> physical = ModuleOp::create(module.getLoc());
  auto options = readCompileOptions(module);
  if (failed(options)) return failure();
  (*physical)->setAttr(compileOptionsAttr, *options);
  kir_to_dsa::Construction construction(module, *physical, configuration, shapes, strides);
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (failed(construction.lower(function))) return failure();
  realizeIntegerStorage(*physical);
  if (failed(dsa::verifyProgram(*physical))) return failure();
  module.getBodyRegion().takeBody(physical->getBodyRegion());
  module->setAttrs((*physical)->getAttrs());
  return success();
}
}
