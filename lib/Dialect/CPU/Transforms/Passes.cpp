#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_REGISTRATION
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

void addNormalization(OpPassManager &manager) {
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
}

} // namespace

void registerCPUPasses() {
  registerCPUTransformPasses();
  PassPipelineRegistration<CPUCompilationOptions>(
      "intent-cpu-pipeline",
      "Construct configured CPU tasks with the selected compiled provider",
      buildCPUPipeline);
}

void buildCPUPipeline(OpPassManager &manager,
                      const CPUCompilationOptions &options) {
  CPUConfigureTargetOptions target;
  target.vectorBits = options.vectorBits;
  target.workers = options.workers;
  target.privateBytes = options.privateBytes;
  target.matrixI8I32 = options.matrixI8I32;
  manager.addPass(createCPUConfigureTarget(target));
  manager.addPass(createCPUBufferizeValues());

  addNormalization(manager);
  manager.addPass(createCPURealizeSliceCollectives());
  manager.addPass(createCPUNormalizeContractions());
  manager.addPass(createCPUFoldUniformComputations());
  manager.addPass(createCPUFuseStructuredComputations());
  manager.addPass(createCPUNormalizeReductions());
  addNormalization(manager);

  CPUMaterializeConfigurationsOptions configurations;
  configurations.provider = options.provider;
  configurations.defaults = options.defaults;
  configurations.overrides = options.overrides;
  manager.addPass(createCPUMaterializeConfigurations(configurations));

  manager.addPass(createCPUGroupRegions());
  CPURealizeRegionsOptions regions;
  regions.provider = options.provider;
  manager.addPass(createCPURealizeRegions(regions));
  addNormalization(manager);
  CPUFoldContractionInputsOptions inputs;
  inputs.provider = options.provider;
  manager.addPass(createCPUFoldContractionInputs(inputs));
  manager.addPass(createCPURealizeHistograms());
  manager.addPass(createCPUFoldUniformComputations());
  manager.addPass(createCPUFuseStructuredComputations());
  addNormalization(manager);

  manager.addPass(createCPUReusePrivateStorage());
  CPUPrepareInputsOptions preparation;
  preparation.provider = options.provider;
  manager.addPass(createCPUPrepareInputs(preparation));
  CPUGroupWorksetsOptions worksets;
  worksets.provider = options.provider;
  manager.addPass(createCPUGroupWorksets(worksets));
  CPUBlockComputationsOptions blocking;
  blocking.provider = options.provider;
  manager.addPass(createCPUBlockComputations(blocking));

  CPUPartitionTasksOptions tasks;
  tasks.provider = options.provider;
  manager.addPass(createCPUPartitionTasks(tasks));
  manager.addPass(createCPUReuseScratchStorage());
  manager.addPass(createCPUOptimizeMemoryAccesses());
  addNormalization(manager);
  manager.addPass(createCPUIsolateTasks());
}

} // namespace intent::cpu
