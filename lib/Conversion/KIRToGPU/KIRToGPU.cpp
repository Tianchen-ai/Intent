#include "Construction.h"
#include "Intent/Conversion/KIRToGPU/KIRToGPU.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Transforms/Passes.h"

using namespace mlir;

namespace intent {

LogicalResult lowerCanonicalKIRToGPU(
    ModuleOp module,
    const GPUCapabilities &capabilities) {
  if (failed(verifyKernelModule(module)))
    return failure();
  if (capabilities.computeUnits <= 0 || capabilities.sharedMemoryPerUnit <= 0 ||
      capabilities.maxDynamicSharedMemoryPerBlock <= 0 ||
      capabilities.registersPerUnit <= 0 ||
      capabilities.maxThreadsPerBlock <= 0 ||
      capabilities.computeCapabilityMajor <= 0 ||
      capabilities.computeCapabilityMinor < 0 ||
      capabilities.singleToDoublePrecisionPerfRatio <= 0)
    return module.emitError("selected GPU capabilities are incomplete");
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1)
    return module.emitError("GPU construction requires exactly one kernel entry");
  if (failed(kir_to_gpu::constructGPUProgram(module, capabilities, functions.front())))
    return failure();
  return gpu::completeGPUProgramConstruction(module);
}

} // namespace intent
