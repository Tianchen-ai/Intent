#include "Intent/Dialect/GPU/Transforms/Mapping/UniformBranches.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Control/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Region/Realization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Reduction/ReductionRealization.h"
#include "Control/TraversalTails.h"
#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace intent::gpu {
#define GEN_PASS_DEF_NORMALIZESTRUCTUREDSOURCESPASS
#define GEN_PASS_DEF_PREDICATESCALARCONTROLPASS
#define GEN_PASS_DEF_POINTWISEOWNERSHIPPASS
#define GEN_PASS_DEF_REDUCTIONCONSUMERSPASS
#define GEN_PASS_DEF_POINTWISEBLOCKINGPASS
#define GEN_PASS_DEF_SCANCONSUMERSPASS
#define GEN_PASS_DEF_RETAINEDVALUESPASS
#define GEN_PASS_DEF_ONLINEREDUCTIONSPASS
#define GEN_PASS_DEF_REGIONFOLDSPASS
#define GEN_PASS_DEF_REGIONSCANSPASS
#define GEN_PASS_DEF_REDUCTIONSPASS
#define GEN_PASS_DEF_CONTRACTIONSPASS
#define GEN_PASS_DEF_ACCESSCOMPOSITIONPASS
#define GEN_PASS_DEF_PRIVATESTORESPASS
#define GEN_PASS_DEF_BUFFERVECTORIZATIONPASS
#define GEN_PASS_DEF_BUFFERPROMOTIONPASS
#define GEN_PASS_DEF_PROGRAMMAPPINGPASS
#define GEN_PASS_DEF_RANGEPREDICATESPASS
#define GEN_PASS_DEF_COMMONVALUESPASS
#define GEN_PASS_DEF_HOISTLOOPINVARIANTVALUESPASS
#define GEN_PASS_DEF_REALIZESHAREDPROGRAMPASS
#define GEN_PASS_DEF_MATERIALIZECONFIGURATIONSPASS
#define GEN_PASS_DEF_FUSEINDEPENDENTTRAVERSALSPASS
#define GEN_PASS_DEF_NORMALIZECOMPLETEDREDUCTIONSPASS
#define GEN_PASS_DEF_PEELTRAVERSALTAILSPASS
#include "Intent/Dialect/GPU/Transforms/Passes.h.inc"

namespace {

// Complete transformation entries own relation closure; the pipeline schedules
// their semantic dependencies and verifies each finished group.
LogicalResult normalizeStructuredSources(ModuleOp module, bool hoist) {
  if (failed(hoist ? hoistLoopInvariantValues(module)
                   : eliminateCommonValues(module)))
    return failure();
  return normalizeContractionSources(module);
}

LogicalResult realizeReductionGroup(ModuleOp module) {
  if (failed(fuseIndependentReductions(module))) return failure();
  return realizeReductionBlocking(module);
}

LogicalResult composeRealizedAccesses(ModuleOp module) {
  if (failed(realizeAccessComposition(module)) ||
      failed(materializeRetainedValues(module)) ||
      failed(realizeAccessComposition(module)))
    return failure();
  return orientLoopContractions(module);
}

LogicalResult simplifyValues(ModuleOp module, bool hoist) {
  if (failed(hoist ? hoistLoopInvariantValues(module)
                   : eliminateCommonValues(module)))
    return failure();
  return simplifyMaskedAccessCoordinates(module);
}

LogicalResult closeSharedConfigurations(func::FuncOp kernel) {
  eraseUnusedParameters(kernel);
  if (failed(materializeSharedConfigTuples(kernel)))
    return failure();
  return verifySharedConfigTuples(kernel);
}

LogicalResult finishTransformation(ModuleOp module, StringRef name,
                                   LogicalResult result) {
  if (failed(result))
    return module.emitError() << "shared GPU transformation failed: " << name;
  if (failed(verifyGPUProgram(module)))
    return module.emitError() << "shared GPU postcondition failed: " << name;
  return success();
}

class NormalizeStructuredSourcesPass : public impl::NormalizeStructuredSourcesPassBase<NormalizeStructuredSourcesPass> {
public:
  using NormalizeStructuredSourcesPassBase::NormalizeStructuredSourcesPassBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(),
                                   normalizeStructuredSources(module, hoistLoopInvariants))))
      signalPassFailure();
  }
};

class PredicateScalarControlPass : public impl::PredicateScalarControlPassBase<PredicateScalarControlPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), predicateScalarControl(module))))
      signalPassFailure();
  }
};

class PointwiseOwnershipPass : public impl::PointwiseOwnershipPassBase<PointwiseOwnershipPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizePointwiseOwnership(module))))
      signalPassFailure();
  }
};

class PointwiseBlockingPass : public impl::PointwiseBlockingPassBase<PointwiseBlockingPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizePointwiseBlocking(module))))
      signalPassFailure();
  }
};

class ScanConsumersPass : public impl::ScanConsumersPassBase<ScanConsumersPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeScanConsumerTraversals(module))))
      signalPassFailure();
  }
};

class RetainedValuesPass : public impl::RetainedValuesPassBase<RetainedValuesPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), materializeRetainedValues(module))))
      signalPassFailure();
  }
};

class OnlineReductionsPass : public impl::OnlineReductionsPassBase<OnlineReductionsPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeOnlineReductions(module))))
      signalPassFailure();
  }
};

class RegionFoldsPass : public impl::RegionFoldsPassBase<RegionFoldsPass> {
public:
  using RegionFoldsPassBase::RegionFoldsPassBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(),
                                   realizeRegionFolds(module, simplifyFirstSummary))))
      signalPassFailure();
  }
};

class RegionScansPass : public impl::RegionScansPassBase<RegionScansPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeRegionScans(module))))
      signalPassFailure();
  }
};

class ReductionsPass : public impl::ReductionsPassBase<ReductionsPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeReductionGroup(module))))
      signalPassFailure();
  }
};

class ContractionsPass : public impl::ContractionsPassBase<ContractionsPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeContractionBlocking(module))))
      signalPassFailure();
  }
};

class AccessCompositionPass : public impl::AccessCompositionPassBase<AccessCompositionPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), composeRealizedAccesses(module))))
      signalPassFailure();
  }
};

class PrivateStoresPass : public impl::PrivateStoresPassBase<PrivateStoresPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), schedulePrivateStores(module))))
      signalPassFailure();
  }
};

class BufferVectorizationPass : public impl::BufferVectorizationPassBase<BufferVectorizationPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), vectorizeBufferLoops(module))))
      signalPassFailure();
  }
};

class BufferPromotionPass : public impl::BufferPromotionPassBase<BufferPromotionPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), promoteBufferValues(module))))
      signalPassFailure();
  }
};

class ProgramMappingPass : public impl::ProgramMappingPassBase<ProgramMappingPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), refineProgramMapping(module))))
      signalPassFailure();
  }
};

class RangePredicatesPass : public impl::RangePredicatesPassBase<RangePredicatesPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), simplifyRangePredicates(module))))
      signalPassFailure();
  }
};

class CommonValuesPass : public impl::CommonValuesPassBase<CommonValuesPass> {
public:
  using CommonValuesPassBase::CommonValuesPassBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(),
                                   simplifyValues(module, hoistLoopInvariants))))
      signalPassFailure();
  }
};

class HoistLoopInvariantValuesPass
    : public impl::HoistLoopInvariantValuesPassBase<HoistLoopInvariantValuesPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyGPUProgram(module)) ||
        failed(finishTransformation(module, getArgument(),
                                    hoistLoopInvariantValues(module))))
      signalPassFailure();
  }
};

class ReductionConsumersPass
    : public impl::ReductionConsumersPassBase<ReductionConsumersPass> {
public:
  using ReductionConsumersPassBase::ReductionConsumersPassBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(),
                                   realizeReductionConsumerTraversals(
                                       module, preferredResidentPrograms))))
      signalPassFailure();
  }
};

void populateTransformations(OpPassManager &manager,
                              unsigned preferredResidentPrograms) {
  manager.addPass(createNormalizeStructuredSourcesPass());
  manager.addPass(createPredicateScalarControlPass());
  manager.addPass(createPointwiseOwnershipPass());
  ReductionConsumersPassOptions consumers;
  consumers.preferredResidentPrograms = preferredResidentPrograms;
  manager.addPass(createReductionConsumersPass(consumers));
  manager.addPass(createPointwiseBlockingPass());
  manager.addPass(createScanConsumersPass());
  manager.addPass(createRetainedValuesPass());
  manager.addPass(createOnlineReductionsPass());
  manager.addPass(createRegionFoldsPass());
  manager.addPass(createRegionScansPass());
  manager.addPass(createReductionsPass());
  manager.addPass(createContractionsPass());
  manager.addPass(createAccessCompositionPass());
  manager.addPass(createPrivateStoresPass());
  manager.addPass(createBufferVectorizationPass());
  manager.addPass(createBufferPromotionPass());
  manager.addPass(createProgramMappingPass());
  manager.addPass(createRangePredicatesPass());
  manager.addPass(createCommonValuesPass());
}

class RealizeSharedProgramPass
    : public impl::RealizeSharedProgramPassBase<RealizeSharedProgramPass> {
public:
  using RealizeSharedProgramPassBase::RealizeSharedProgramPassBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto transform = [&]() -> LogicalResult {
      if (failed(verifyGPUProgram(module))) return failure();
      auto kernel = getPhysicalKernel(module);
      if (failed(kernel)) return failure();
      OpPassManager pipeline(ModuleOp::getOperationName());
      populateTransformations(pipeline, preferredResidentPrograms);
      if (scf::IfOp conditional = independentUniformBranches(*kernel)) {
        auto realized = realizeUniformBranches(module, *kernel, conditional,
            [&](ModuleOp branch) { return runPipeline(pipeline, branch); });
        if (failed(realized)) return failure();
        if (*realized) return success();
      }
      return runPipeline(pipeline, module);
    };
    if (failed(finishTransformation(module, getArgument(), transform())))
      signalPassFailure();
  }
};

class MaterializeConfigurationsPass
    : public impl::MaterializeConfigurationsPassBase<MaterializeConfigurationsPass> {
public:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto kernel = getPhysicalKernel(module);
    if (failed(kernel) || failed(finishTransformation(module, getArgument(),
                                        closeSharedConfigurations(*kernel))))
      signalPassFailure();
  }
};

class FuseIndependentTraversalsPass
    : public impl::FuseIndependentTraversalsPassBase<FuseIndependentTraversalsPass> {
public:
  using FuseIndependentTraversalsPassBase::FuseIndependentTraversalsPassBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto transform = [&]() -> LogicalResult {
      if (failed(fuseIndependentTraversals(module, hoistLoopInvariants)))
        return failure();
      auto kernel = getPhysicalKernel(module);
      if (failed(kernel)) return failure();
      return verifySharedConfigTuples(*kernel);
    };
    if (failed(finishTransformation(module, getArgument(), transform())))
      signalPassFailure();
  }
};

class NormalizeCompletedReductionsPass
    : public impl::NormalizeCompletedReductionsPassBase<NormalizeCompletedReductionsPass> {
public:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto transform = [&]() -> LogicalResult {
      if (failed(verifyGPUProgram(module))) return failure();
      auto kernel = getPhysicalKernel(module);
      if (failed(kernel) || failed(verifySharedConfigTuples(*kernel)))
        return failure();
      if (reduction::normalizeCompletedReductions(*kernel)) {
        eraseDeadPhysicalValues(*kernel);
        if (failed(closeValueRelations(*kernel))) return failure();
      }
      return verifySharedConfigTuples(*kernel);
    };
    if (failed(finishTransformation(module, getArgument(), transform())))
      signalPassFailure();
  }
};

class PeelTraversalTailsPass
    : public impl::PeelTraversalTailsPassBase<PeelTraversalTailsPass> {
public:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto transform = [&]() -> LogicalResult {
      if (failed(verifyGPUProgram(module))) return failure();
      auto kernel = getPhysicalKernel(module);
      if (failed(kernel) || failed(verifySharedConfigTuples(*kernel)))
        return failure();
      if (peelTraversalTails(*kernel)) {
        if (failed(simplifyRangePredicates(module)) ||
            failed(eliminateCommonValues(module)))
          return failure();
        eraseDeadPhysicalValues(*kernel);
        if (failed(closeValueRelations(*kernel))) return failure();
      }
      return verifySharedConfigTuples(*kernel);
    };
    if (failed(finishTransformation(module, getArgument(), transform())))
      signalPassFailure();
  }
};

} // namespace

#define GEN_PASS_REGISTRATION
#include "Intent/Dialect/GPU/Transforms/Passes.h.inc"

void registerGPUPasses() {
  registerIntentGPUTransformPasses();
  PassPipelineRegistration<>("intent-gpu-shared",
      "Realize and close the shared executable GPU program",
      [](OpPassManager &manager) { buildSharedGPUPipeline(manager); });
}

LogicalResult completeGPUProgramConstruction(ModuleOp module) {
  // Construction closes the initial indexed access graph. Ownership/blocking
  // and structured realization are subsequent transformations of this program.
  if (failed(realizeAccessComposition(module)))
    return failure();
  return verifyGPUProgram(module);
}

void buildSharedGPUPipeline(OpPassManager &manager,
                            unsigned preferredResidentPrograms) {
  RealizeSharedProgramPassOptions options;
  options.preferredResidentPrograms = preferredResidentPrograms;
  manager.addPass(createRealizeSharedProgramPass(options));
  manager.addPass(createMaterializeConfigurationsPass());
  manager.addPass(createFuseIndependentTraversalsPass());
  manager.addPass(createNormalizeCompletedReductionsPass());
  manager.addPass(createPeelTraversalTailsPass());
}

} // namespace intent::gpu
