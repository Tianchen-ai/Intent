#include "Intent/Target/CuTile/Transforms/Access/ArrayViews.h"
#include "Intent/Target/CuTile/Transforms/Compute/MMALoops.h"
#include "Intent/Target/CuTile/Transforms/Control/Loops.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Target/CuTile/Serialization/Serializer.h"
#include "Configuration/Configurations.h"
#include "Compute/StrictArithmetic.h"
#include "Access/FragmentStorage.h"
#include "Program.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "Intent/Dialect/GPU/Transforms/Mapping/ExecutionGroups.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Contraction/Contraction.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"

using namespace mlir;
namespace intent::cutile {

LogicalResult prepareProgram(ModuleOp module) {
  if (failed(gpu::verifyGPUProgram(module)) ||
      failed(gpu::lowerExecutionGroups(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  legalizeStrictArithmetic(*kernel);
  if (failed(gpu::contraction::normalizeMatrixContractShapes(*kernel)) ||
      failed(gpu::verifyGPUProgram(module)))
    return failure();
  if (failed(prepareLaunchConfigurations(*kernel)) ||
      failed(materializeFragmentStorage(*kernel)) ||
      failed(gpu::lowerWorkspaceAllocations(module)))
    return failure();
  gpu::foldExactConstantDivisions(*kernel);
  return gpu::verifyGPUProgram(module);
}

LogicalResult finalizeProgram(ModuleOp module, bool hoistLoopInvariants) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel)) return failure();
  if (failed(refineMMALoops(module)))
    return failure();
  if (failed(materializeClosedConfigs(*kernel)))
    return failure();
  gpu::foldScalarIntegerValues(*kernel);
  realizeWideLoops(*kernel);
  preserveNativeIndexValues(*kernel);
  if (failed(collapseArrayViews(module)))
    return failure();
  if (failed(hoistLoopInvariants ? gpu::hoistLoopInvariantValues(module)
                                : gpu::eliminateCommonValues(module)))
    return failure();
  if (failed(finalizeConfigurationRequirements(*kernel)))
    return failure();
  if (failed(gpu::projectBoundedParameterUses(*kernel)))
    return failure();
  if (auto bounds = arrayIndexTileBounds(*kernel)) {
    (*kernel)->setAttr(arrayIndexTileBoundsAttr, UnitAttr::get(module.getContext()));
    for (auto [resource, shape] : *bounds) {
      if (auto argument = dyn_cast<BlockArgument>(resource))
        kernel->setArgAttr(argument.getArgNumber(), arrayIndexTileBoundsAttr, shape);
      else
        resource.getDefiningOp()->setAttr(arrayIndexTileBoundsAttr, shape);
    }
  }
  if (failed(verifyCuTileProgram(module, verifySourceOperation)))
    return failure();
  return success();
}
} // namespace intent::cutile
