#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "Intent/Transforms/Components.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_REGISTRATION
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

void addNormalization(OpPassManager &manager) {
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
}

using ComponentVisitor = llvm::function_ref<void(TransformationComponent, std::unique_ptr<Pass>)>;

void visitValueComponents(ComponentVisitor add, const CPUCompilationOptions &options) {
  CPUConfigureTargetOptions target;
  target.target = options.target;
  add({"cpu", "values", true}, createCPUConfigureTarget(target));
  CPUBindProfilesOptions profiles;
  profiles.provider = options.provider;
  profiles.defaults = options.defaults;
  profiles.overrides = options.overrides;
  add({"cpu", "values", true}, createCPUBindProfiles(profiles));
  add({"cpu", "values", false}, createCPUFuseTensorComputations());
}

void visitBufferComponents(ComponentVisitor add, llvm::function_ref<void()> normalize,
                           const CPUCompilationOptions &options) {
  auto required = [&](StringRef stage, std::unique_ptr<Pass> pass) { add({"cpu", stage, true}, std::move(pass)); };
  auto optimize = [&](StringRef stage, std::unique_ptr<Pass> pass) { add({"cpu", stage, false}, std::move(pass)); };
  required("buffers", createCPURealizeSliceCollectives());
  required("buffers", createCPUNormalizeContractions());
  optimize("buffers", createCPUFoldUniformComputations());
  optimize("buffers", createCPUFuseStructuredComputations());
  required("buffers", createCPUNormalizeReductions());
  normalize();
  CPUMaterializeConfigurationsOptions configurations;
  configurations.provider = options.provider;
  required("selection", createCPUMaterializeConfigurations(configurations));
  optimize("regions", createCPUGroupRegions());
  CPURealizeRegionsOptions regions;
  regions.provider = options.provider;
  required("regions", createCPURealizeRegions(regions));
  normalize();
  CPUFoldContractionInputsOptions inputs;
  inputs.provider = options.provider;
  optimize("supply", createCPUFoldContractionInputs(inputs));
  required("buffers", createCPURealizeHistograms());
  optimize("buffers", createCPUFoldUniformComputations());
  optimize("buffers", createCPUFuseStructuredComputations());
  normalize();
  optimize("storage", createCPUReusePrivateStorage());
  CPUPrepareInputsOptions preparation;
  preparation.provider = options.provider;
  required("supply", createCPUPrepareInputs(preparation));
  CPUGroupWorksetsOptions worksets;
  worksets.provider = options.provider;
  required("tasks", createCPUGroupWorksets(worksets));
  CPUBlockComputationsOptions blocking;
  blocking.provider = options.provider;
  required("tasks", createCPUBlockComputations(blocking));
  CPUPartitionTasksOptions tasks;
  tasks.provider = options.provider;
  required("tasks", createCPUPartitionTasks(tasks));
  optimize("storage", createCPUReuseScratchStorage());
  optimize("storage", createCPUOptimizeMemoryAccesses());
  normalize();
  required("tasks", createCPUIsolateTasks());
  optimize("tasks", createCPUSpecializeSingletonTasks());
}

} // namespace

void registerCPUPasses() {
  registerCPUTransformPasses();
  CPUCompilationOptions options;
  visitValueComponents(registerTransformation, options);
  visitBufferComponents(registerTransformation, [] {}, options);
  registerTransformation({"cpu", "materialization", true}, createCPUBufferizeValues());
  PassPipelineRegistration<CPUCompilationOptions>(
      "intent-cpu-values", "Bind the target and normalize CPU SSA values", buildCPUValuePipeline);
  PassPipelineRegistration<CPUCompilationOptions>(
      "intent-cpu-buffers", "Optimize and configure an already materialized CPU program", buildCPUBufferPipeline);
}

void buildCPUValuePipeline(OpPassManager &manager,
                          const CPUCompilationOptions &options) {
  addNormalization(manager);
  visitValueComponents([&](TransformationComponent component, std::unique_ptr<Pass> pass) {
    addTransformation(manager, component, std::move(pass));
  }, options);
}

void buildCPUBufferPipeline(OpPassManager &manager,
                           const CPUCompilationOptions &options) {
  addNormalization(manager);
  visitBufferComponents([&](TransformationComponent component, std::unique_ptr<Pass> pass) {
    addTransformation(manager, component, std::move(pass));
  }, [&] { addNormalization(manager); }, options);
}

} // namespace intent::cpu
