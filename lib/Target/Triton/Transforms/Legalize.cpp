#include "Program.h"
#include "Access/AccessForms.h"
#include "Access/Gathers.h"
#include "Access/Scatter.h"
#include "Collective/Collectives.h"
#include "Configuration/Configurations.h"
#include "Supply/Supply.h"
#include "Value/Values.h"
#include "Intent/Target/Triton/Analysis/Program.h"
#include "Intent/Target/Triton/Serialization/Serializer.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Transforms/Contraction/Contraction.h"
#include "Intent/Dialect/Block/Transforms/Mapping/ExecutionGroups.h"
#include "Intent/Dialect/Block/Transforms/Passes.h"
#include "Intent/Dialect/Block/Transforms/Storage/Workspace.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueMaterialization.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::triton {

LogicalResult prepareTritonMemory(ModuleOp module) {
  if (failed(block::verifyBlockProgram(module)) ||
      failed(block::realizeShortWindowReductions(module)) ||
      failed(block::lowerExecutionGroups(module)))
    return failure();
  auto kernel = block::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  detail::foldIntegerScanTails(*kernel);
  if (failed(detail::materializeGatherSources(*kernel)) ||
      failed(detail::legalizeLargeScalarGathers(*kernel)) ||
      failed(detail::legalizeGatherCoordinates(*kernel)) ||
      failed(detail::legalizeExpandingGathers(*kernel)) ||
      failed(detail::legalizeScatterAdd(*kernel)) ||
      failed(block::contraction::normalizeMatrixContractShapes(*kernel)))
    return failure();
  return block::verifyBlockProgram(module);
}

LogicalResult formTritonProgram(ModuleOp module) {
  auto kernel = block::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  bool requiresCtaSynchronization = detail::hasOrderedViewDependencies(*kernel);
  kernel->walk([&](block::BufferOp) { requiresCtaSynchronization = true; });
  auto localOptions = declareProviderOptions(
      *kernel, requiresCtaSynchronization,
      detail::findLoadPipelineLoops(*kernel));
  if (failed(localOptions) || failed(block::verifyBlockProgram(module)) ||
      failed(block::lowerWorkspaceAllocations(module)))
    return failure();
  if (failed(detail::legalizeOrderedViewDependencies(*kernel)) ||
      failed(detail::legalizeSplitGatherPairs(*kernel)))
    return failure();
  detail::selectRecurrencePipelineStages(*kernel);
  detail::selectOrderedLoadUnrolling(*kernel);
  auto descriptors =
      detail::materializeTensorDescriptorForms(*kernel, *localOptions);
  if (failed(descriptors))
    return failure();
  if (failed(detail::orientPointerLoads(*kernel)))
    return failure();
  if (failed(materializeLegalConfigs(*kernel, *descriptors)))
    return failure();
  detail::selectContractForms(*kernel);
  // Native extensions now belong to the current program; the shared surface
  // verifier is no longer applicable.
  return mlir::verify(module);
}

LogicalResult finalizeTritonProgram(ModuleOp module, bool hoistLoopInvariants) {
  auto kernel = block::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  block::foldExactConstantDivisions(*kernel);
  block::narrowIntegerArithmetic(*kernel);
  detail::canonicalizeBroadcastProjections(*kernel);
  SmallVector<block::AssumeInBoundsOp> assumptions;
  kernel->walk([&](block::AssumeInBoundsOp assumption) {
    assumptions.push_back(assumption);
  });
  for (block::AssumeInBoundsOp assumption : assumptions)
    assumption.erase();
  if (failed(hoistLoopInvariants ? block::hoistLoopInvariantValues(module)
                                : block::eliminateCommonValues(module)) ||
      failed(detail::legalizeCollectiveCallbacks(*kernel)) ||
      failed(finalizeConfigurationRequirements(*kernel)))
    return failure();
  detail::sinkSelectProducers(*kernel);
  return verifyTritonProgram(module, verifySourceOperation);
}

} // namespace intent::triton
