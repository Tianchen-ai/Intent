#include "Intent/Conversion/KIRToCPU/KIRToCPU.h"
#include "Construction.h"
#include "Intent/Conversion/ScalarLowering.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent {

LogicalResult lowerCanonicalKIRToCPU(ModuleOp module, CPUEntryLayout entryLayout) {
  CanonicalKernelAnalysis analysis(module);
  if (failed(analysis.verify())) return failure();
  auto physical = OwningOpRef<ModuleOp>(ModuleOp::create(module.getLoc()));
  auto options = readCompileOptions(module);
  if (failed(options)) return failure();
  (*physical)->setAttr(compileOptionsAttr, *options);
  kir_to_cpu::Construction construction(analysis, *physical, entryLayout);
  unsigned count = 0;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (++count != 1)
      return function.emitError("CPU construction requires an inlined single kernel");
    if (failed(construction.lower(function))) return failure();
  }
  if (count == 0) return module.emitError("CPU construction found no kernel");
  realizeIntegerStorage(*physical);
  if (failed(mlir::verify(*physical))) return failure();
  module.getBodyRegion().takeBody(physical->getBodyRegion());
  module->setAttrs((*physical)->getAttrs());
  return success();
}

}
