#include "Intent/Target/DSA/BangC/Physical/Transforms/Passes.h"
#include "Intent/Transforms/Components.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace intent::bangc::physical {

void buildDSAPipeline(OpPassManager &manager) {
  addTransformation(manager, {"bangc", "physical", true}, createDSARealizeCollectives());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
  addTransformation(manager, {"bangc", "physical", false}, createDSASimplifyWorksetBounds());
  addTransformation(manager, {"bangc", "physical", false}, createDSAComposePrivateVersions());
  addTransformation(manager, {"bangc", "physical", false}, createDSACollectiveGatherSupply());
  addTransformation(manager, {"bangc", "physical", false}, createDSAMatrixSupply());
  addTransformation(manager, {"bangc", "physical", false}, createDSANormalizeIndices());
}

void registerDSAPipelines() {
  registerTransformation({"bangc", "physical", true}, createDSARealizeCollectives());
  registerTransformation({"bangc", "physical", false}, createDSASimplifyWorksetBounds());
  registerTransformation({"bangc", "physical", false}, createDSAComposePrivateVersions());
  registerTransformation({"bangc", "physical", false}, createDSACollectiveGatherSupply());
  registerTransformation({"bangc", "physical", false}, createDSAMatrixSupply());
  registerTransformation({"bangc", "physical", false}, createDSANormalizeIndices());
  PassPipelineRegistration<>(
      "intent-dsa", "Form a complete shared DSA program",
      [](OpPassManager &manager) { buildDSAPipeline(manager); });
}

} // namespace intent::bangc::physical
