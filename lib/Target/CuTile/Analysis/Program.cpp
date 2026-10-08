#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Intent/Target/CuTile/Analysis/Configuration.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "Intent/Target/CuTile/IR/Configuration.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
namespace intent::cutile {

bool supportsE8M0ScaledMMA(gpu::CapabilitiesAttr capabilities) {
  return capabilities && capabilities.getComputeCapabilityMajor() >= 10;
}

void NativeProgramFeatures::observe(Operation *operation) {
  bool matrix = isa<gpu::ContractOp, gpu::ScaledContractOp, MMAOp,
                    ScaledMMAOp>(operation);
  matrixCompute |= matrix;
  occupancySensitive |= matrix ||
      isa<gpu::ReduceOp, gpu::HistogramOp, gpu::ScanOp, ReduceOp, ScanOp>(operation);
}

NativeProgramFeatures queryNativeProgramFeatures(func::FuncOp kernel) {
  NativeProgramFeatures features;
  kernel.walk([&](Operation *operation) { features.observe(operation); });
  return features;
}

namespace {
bool isCuTileScalarType(Type type) {
  if (type.isIndex() ||
      isa<Float16Type, BFloat16Type, Float32Type, Float64Type, Float8E4M3FNType,
          Float8E5M2Type>(type))
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.getWidth() == 1 || integer.getWidth() == 8 ||
                     integer.getWidth() == 16 || integer.getWidth() == 32 ||
                     integer.getWidth() == 64);
}

LogicalResult verifyKernel(func::FuncOp kernel,
    llvm::function_ref<LogicalResult(Operation *)> verifySourceOperation) {
  if (kernel->hasAttr(arrayIndexTileBoundsAttr)) {
    auto expected = arrayIndexTileBounds(kernel);
    if (!kernel->getAttrOfType<UnitAttr>(arrayIndexTileBoundsAttr) || !expected)
      return kernel.emitError("cuTile array-index bounds do not cover the current native accesses");
    for (auto [resource, bounds] : *expected)
      if (getNativeArrayIndexBounds(resource) != bounds)
        return kernel.emitError("cuTile array-index bounds do not match their current native array");
  }
  for (BlockArgument argument : kernel.getArguments())
    if (kernel.getArgAttr(argument.getArgNumber(), arrayIndexTileBoundsAttr) &&
        (!kernel->hasAttr(arrayIndexTileBoundsAttr) || !isa<gpu::ViewType>(argument.getType())))
      return kernel.emitError("cuTile array-index bounds require a selected view binding");
  auto arrayBounds = kernel.walk([&](ArrayViewOp array) {
    if (array->hasAttr(arrayIndexTileBoundsAttr) && !kernel->hasAttr(arrayIndexTileBoundsAttr)) {
      array.emitOpError("array-index bounds require a selected native index width binding");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (arrayBounds.wasInterrupted())
    return failure();
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.size() != 1)
    return kernel.emitError(
        "cuTile provider currently requires one explicit linear program space");
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError("cuTile provider requires selected GPU capabilities");
  auto parameters = gpu::ParameterSpace::read(kernel);
  if (failed(parameters)) return failure();
  gpu::ParameterAttr accessForm, occupancy, ctas, workerWarps, loadPolicy;
  for (gpu::ParameterAttr schema : parameters->declarations()) {
    auto role = schema.getRole();
    bool provider = schema.getPhase() == gpu::ConfigurationBindingPhase::Provider;
    if (!schema.isExtent() || provider != isCuTileProviderRole(role))
      return kernel.emitError("cuTile program contains a foreign provider parameter");
    if (!provider)
      continue;
    ArrayRef<int64_t> candidates = schema.getCandidates().asArrayRef();
    gpu::ParameterAttr *binding = nullptr;
    StringRef name;
    bool (*legal)(int64_t) = nullptr;
    if (role == gpu::ParameterRole::ProviderAccessForm) {
      binding = &accessForm; name = accessFormParameter; legal = isLegalAccessForm;
    } else if (role == gpu::ParameterRole::ProviderCTAs) {
      binding = &ctas; name = ctasParameter; legal = isLegalCTAs;
    } else if (role == gpu::ParameterRole::ProviderWarps) {
      binding = &workerWarps; name = workerWarpsParameter; legal = isLegalWorkerWarps;
    } else if (role == gpu::ParameterRole::ProviderLoadPolicy) {
      binding = &loadPolicy; name = loadPolicyParameter; legal = isLegalLoadPolicy;
    } else {
      binding = &occupancy; name = occupancyParameter; legal = isLegalOccupancy;
    }
    if (*binding || schema.getName().getValue() != name ||
        candidates.empty() || !llvm::all_of(candidates, legal))
      return kernel.emitError("invalid or duplicate cuTile provider declaration: ") << schema.getName();
    *binding = schema;
  }
  auto parameterUses = kernel.walk([&](gpu::ParameterOp parameter) -> WalkResult {
    auto schema = parameter.getDeclaration();
    auto role = schema.getRole();
    if ((role == gpu::ParameterRole::ProviderCTAs ||
         role == gpu::ParameterRole::ProviderWarps ||
         role == gpu::ParameterRole::ProviderOccupancy) &&
        !parameter.getResult().use_empty()) {
      parameter.emitOpError("cuTile compiler hints cannot be read by the kernel body");
      return WalkResult::interrupt();
    }
    if (role == gpu::ParameterRole::ProviderLoadPolicy)
      for (OpOperand &use : parameter.getResult().getUses()) {
        auto load = dyn_cast<TileLoadOp>(use.getOwner());
        auto gather = dyn_cast<GatherLoadOp>(use.getOwner());
        if ((!load || load.getLatencyPolicy() != parameter.getResult()) &&
            (!gather || gather.getLatencyPolicy() != parameter.getResult())) {
          parameter.emitOpError(
              "cuTile load latency must bind a tile or gather load");
          return WalkResult::interrupt();
        }
      }
    return WalkResult::advance();
  });
  if (parameterUses.wasInterrupted())
    return failure();
  if (failed(verifyClosedConfigs(kernel)))
    return failure();
  const NativeProgramFeatures features = queryNativeProgramFeatures(kernel);
  if (features.occupancySensitive != static_cast<bool>(occupancy))
    return kernel.emitError(
        "cuTile occupancy hint does not match occupancy-sensitive tile compute");
  if (features.matrixCompute != static_cast<bool>(ctas))
    return kernel.emitError("cuTile CTA hint does not match matrix tile compute");
  auto isNativeTMACondition = [&](Value value) {
    auto compare = value.getDefiningOp<gpu::CompareOp>();
    return accessForm && compare &&
           compare.getPredicate() == ComparePredicate::Ne &&
           gpu::queryParameter(compare.getLhs()) == accessForm &&
           gpu::IndexRelations().constant(compare.getRhs()) == nativeNoTMAForm;
  };
  WalkResult result = kernel.walk([&](Operation *operation) {
    if (isa<gpu::LoadOp, gpu::StoreOp, gpu::ContractOp>(operation)) {
      operation->emitOpError(
          "was not converted to an explicit cuTile tile/MMA form");
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<TileLoadOp>(operation)) {
      if (!isNativeTMACondition(load.getAllowTma())) {
        load.emitOpError(
            "allow_tma is not the typed cuTile access-form decision");
        return WalkResult::interrupt();
      }
    }
    if (auto store = dyn_cast<TileStoreOp>(operation)) {
      if (!isNativeTMACondition(store.getAllowTma())) {
        store.emitOpError(
            "allow_tma is not the typed cuTile access-form decision");
        return WalkResult::interrupt();
      }
    }
    if (auto scaled = dyn_cast<ScaledMMAOp>(operation)) {
      if (!supportsE8M0ScaledMMA(capabilities)) {
        scaled.emitOpError(
            "requires compute capability 10.0 or newer for E8M0 scaled MMA");
        return WalkResult::interrupt();
      }
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation))
      if (!loop.getInductionVar().getType().isSignlessInteger(32)) {
        loop.emitOpError("cuTile native for requires an i32 induction variable");
        return WalkResult::interrupt();
      }
    for (Type type : operation->getResultTypes()) {
      if (auto fragment = dyn_cast<gpu::FragmentType>(type)) {
        if (!isCuTileScalarType(fragment.getElementType())) {
          operation->emitOpError("contains a fragment dtype outside cuTile");
          return WalkResult::interrupt();
        }
      } else if (!isa<gpu::ViewType, gpu::RangeType, gpu::RecordType>(type) &&
                 !isCuTileScalarType(type)) {
        operation->emitOpError("has a result type outside the cuTile surface");
        return WalkResult::interrupt();
      }
    }
    return succeeded(verifySourceOperation(operation))
        ? WalkResult::advance() : WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace

LogicalResult verifyCuTileProgram(ModuleOp module,
    llvm::function_ref<LogicalResult(Operation *)> verifySourceOperation) {
  if (failed(mlir::verify(module))) return failure();
  auto kernel = gpu::getPhysicalKernel(module);
  return failed(kernel) ? failure() : verifyKernel(*kernel, verifySourceOperation);
}

} // namespace intent::cutile
