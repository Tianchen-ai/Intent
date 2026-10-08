#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Target/CuTile/Serialization/Serializer.h"
#include "Intent/Target/CuTile/Transforms/Passes.h"
#include "Intent/Target/CuTile/Transforms/Configuration/TuningProfiles.h"
#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "Intent/Target/Triton/Serialization/Serializer.h"
#include "Intent/Target/Triton/Transforms/Passes.h"
#include "Intent/Target/Triton/Transforms/Configuration/TuningProfiles.h"

using namespace mlir;

namespace intent::compiler {

ArrayRef<Backend> gpuBackends() {
  static const Backend adapters[] = {
      {Provider::Triton, "triton", true,
       GPUBackend{true, false, true, 1, triton::tuningProfileSchema(), "triton.json"},
       [](DialectRegistry &registry) { registry.insert<triton::IntentTritonDialect>(); },
       triton::registerTritonPasses,
       [](OpPassManager &manager, const Request &) { triton::buildTritonPipeline(manager); },
       triton::serializeProgram},
      {Provider::CuTile, "cutile", true,
       GPUBackend{true, true, false, 2, cutile::tuningProfileSchema(), "cutile.json"},
       [](DialectRegistry &registry) { registry.insert<cutile::IntentCuTileDialect>(); },
       cutile::registerCuTilePasses,
       [](OpPassManager &manager, const Request &) { cutile::buildCuTilePipeline(manager); },
       cutile::serializeProgram}};
  return adapters;
}

void GPUBackend::buildConstruction(OpPassManager &manager, const Request &request) const {
  ConstructGPUOptions options;
  options.computeUnits = request.gpu.computeUnits;
  options.sharedMemoryPerUnit = request.gpu.sharedMemoryPerUnit;
  options.maxDynamicSharedMemoryPerBlock = request.gpu.maxDynamicSharedMemoryPerBlock;
  options.registersPerUnit = request.gpu.registersPerUnit;
  options.maxThreadsPerBlock = request.gpu.maxThreadsPerBlock;
  options.computeCapabilityMajor = request.gpu.computeCapabilityMajor;
  options.computeCapabilityMinor = request.gpu.computeCapabilityMinor;
  options.singleToDoublePrecisionPerfRatio = request.gpu.singleToDoublePrecisionPerfRatio;
  options.matrixUnits = request.gpu.matrixUnits;
  options.dynamicVectorWidth = request.gpu.dynamicVectorWidth;
  options.nativeTupleReductions = nativeTupleReductions;
  options.nativeTupleReductionRequiresConstantIdentity =
      nativeTupleReductionRequiresConstantIdentity;
  options.nativeFragmentGather = nativeFragmentGather;
  manager.addPass(createConstructGPU(options));
}

void GPUBackend::buildShared(OpPassManager &manager, const Request &request, StringRef provider) const {
  ResolveGPUProfilesOptions options;
  options.directory = request.profileDirectory;
  options.overrides = request.tuningConfig;
  options.provider = provider.str();
  manager.addPass(createResolveGPUProfiles(options));
  gpu::buildSharedGPUPipeline(manager, preferredReductionResidentPrograms);
}

LogicalResult GPUBackend::verifySharedInput(ModuleOp module, const Request &request, StringRef provider) const {
  if (failed(gpu::verifyGPUProgram(module))) return failure();
  auto kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel) || failed(gpu::verifySharedConfigTuples(*kernel))) return failure();
  const auto &device = request.gpu;
  auto expected = gpu::CapabilitiesAttr::getChecked([&] { return module.emitError(); },
      module.getContext(), device.computeUnits, device.sharedMemoryPerUnit,
      device.maxDynamicSharedMemoryPerBlock, device.registersPerUnit,
      device.maxThreadsPerBlock, device.computeCapabilityMajor,
      device.computeCapabilityMinor, device.singleToDoublePrecisionPerfRatio,
      device.matrixUnits, device.dynamicVectorWidth, nativeTupleReductions,
      nativeTupleReductionRequiresConstantIdentity,
      nativeFragmentGather);
  if (!expected) return failure();
  if ((*kernel)->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr) != expected)
    return module.emitError("shared GPU capabilities disagree with the requested device or provider primitive contract");
  auto resolved = gpu::TuningProfiles::from(module);
  if (failed(resolved)) return failure();
  auto tables = module->getAttrOfType<gpu::TuningProfilesAttr>(gpu::tuningProfilesAttr);
  if (tables.getSpaces().size() != 1 || !tables.getSpaces().get(provider))
    return module.emitError("shared GPU configurations belong to a different provider");
  for (const auto &schema : {profiles}) {
    auto table = tables.getSpaces().getAs<gpu::TuningProfileTableAttr>(schema.space);
    if (!table)
      return module.emitError("shared GPU program is missing the resolved profile namespace '")
          << schema.space << "'";
    for (NamedAttribute family : table.getFamilies())
      if (failed(resolved->get(schema, family.getName().getValue(), module.getLoc())))
        return failure();
  }
  return success();
}

} // namespace intent::compiler
