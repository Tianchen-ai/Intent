#include "Construction.h"
#include "Intent/Conversion/ScalarLowering.h"

using namespace mlir;
namespace intent {

LogicalResult lowerCanonicalKIRToDSA(ModuleOp module, bangc::physical::ConfigurationAttr configuration,
                                   DictionaryAttr shapes, DictionaryAttr strides) {
  CanonicalKernelAnalysis analysis(module);
  if (failed(analysis.verify())) return failure();
  OwningOpRef<ModuleOp> physical = ModuleOp::create(module.getLoc());
  auto options = readCompileOptions(module);
  if (failed(options)) return failure();
  (*physical)->setAttr(compileOptionsAttr, *options);
  bangc::construction::Construction construction(module, *physical, configuration, shapes, strides);
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (failed(construction.lower(function))) return failure();
  realizeIntegerStorage(*physical);
  if (failed(bangc::physical::verifyProgram(*physical))) return failure();
  module.getBodyRegion().takeBody(physical->getBodyRegion());
  module->setAttrs((*physical)->getAttrs());
  return success();
}
}
