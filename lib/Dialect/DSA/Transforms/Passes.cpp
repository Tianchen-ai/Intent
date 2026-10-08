#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace intent::dsa {

void buildDSAPipeline(OpPassManager &manager) {
  manager.addPass(createDSARealizeCollectives());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
  manager.addPass(createDSACollectiveGatherSupply());
  manager.addPass(createDSAMatrixSupply());
  manager.addPass(createDSANormalizeIndices());
}

void registerDSAPipelines() {
  PassPipelineRegistration<>(
      "intent-dsa", "Form a complete shared DSA program",
      [](OpPassManager &manager) { buildDSAPipeline(manager); });
}

} // namespace intent::dsa
