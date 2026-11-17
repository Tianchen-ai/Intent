#include "Intent/Dialect/Block/Transforms/Mapping/UniformBranches.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/Block/Transforms/Control/Predication.h"
#include "Intent/Dialect/Block/Transforms/Configuration/TuningProfiles.h"
#include "Intent/Dialect/Block/Transforms/Passes.h"
#include "Intent/Dialect/Block/Transforms/Region/Realization.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueRelations.h"
#include "Reduction/ReductionRealization.h"
#include "Control/TraversalTails.h"
#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "Intent/Transforms/Components.h"

using namespace mlir;

namespace intent::block {
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
#define GEN_PASS_DEF_VECTORCONTRACTIONSPASS
#define GEN_PASS_DEF_CONTRACTIONSPASS
#define GEN_PASS_DEF_ACCESSCOMPOSITIONPASS
#define GEN_PASS_DEF_PRIVATESTORESPASS
#define GEN_PASS_DEF_BUFFERVECTORIZATIONPASS
#define GEN_PASS_DEF_STATICVIEWVALUESPASS
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
#include "Intent/Dialect/Block/Transforms/Passes.h.inc"

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
  auto options = readCompileOptions(module);
  if (failed(options) || failed(realizeAccessComposition(module))) return failure();
  auto enabled = options->getOptimizations().getAs<BoolAttr>("intent-block-materialize-retained-values");
  if ((!enabled || enabled.getValue()) && failed(materializeRetainedValues(module))) return failure();
  if (failed(realizeAccessComposition(module))) return failure();
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
    return module.emitError() << "shared Block transformation failed: " << name;
  if (failed(verifyBlockProgram(module)))
    return module.emitError() << "shared Block postcondition failed: " << name;
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

class VectorContractionsPass : public impl::VectorContractionsPassBase<VectorContractionsPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    auto transform = [&]() -> LogicalResult {
      auto kernel = getPhysicalKernel(module);
      if (failed(kernel) || failed(realizeVectorContractions(module))) return failure();
      return closeValueRelations(*kernel);
    };
    if (failed(finishTransformation(module, getArgument(), transform())))
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

class StaticViewValuesPass : public impl::StaticViewValuesPassBase<StaticViewValuesPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), promoteStaticViewValues(module))))
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
    if (failed(verifyBlockProgram(module)) ||
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

using ComponentVisitor = llvm::function_ref<void(TransformationComponent, std::unique_ptr<Pass>)>;

void visitBlockRealization(ComponentVisitor add, unsigned preferredResidentPrograms) {
  const TransformationComponent source{"block", "source", true};
  const TransformationComponent ownership{"block", "ownership", true};
  const TransformationComponent computation{"block", "computation", true};
  const TransformationComponent computeOptimization{"block", "computation", false};
  const TransformationComponent storage{"block", "storage", true};
  const TransformationComponent storageOptimization{"block", "storage", false};
  const TransformationComponent mapping{"block", "mapping", true};
  const TransformationComponent simplification{"block", "mapping", false};
  add(source, createNormalizeStructuredSourcesPass());
  add(source, createPredicateScalarControlPass());
  add(ownership, createPointwiseOwnershipPass());
  ReductionConsumersPassOptions consumers;
  consumers.preferredResidentPrograms = preferredResidentPrograms;
  add(ownership, createReductionConsumersPass(consumers));
  add(ownership, createPointwiseBlockingPass());
  add(computation, createScanConsumersPass());
  add(computeOptimization, createRetainedValuesPass());
  add(computeOptimization, createOnlineReductionsPass());
  add(computation, createRegionFoldsPass());
  add(computation, createRegionScansPass());
  // Predicate specialization exposes unmasked multiply-reductions.
  add(source, createNormalizeStructuredSourcesPass());
  add(computation, createVectorContractionsPass());
  add(computation, createReductionsPass());
  add(computation, createContractionsPass());
  add(storage, createAccessCompositionPass());
  add(storage, createPrivateStoresPass());
  add(storageOptimization, createStaticViewValuesPass());
  add(storageOptimization, createBufferVectorizationPass());
  add(storageOptimization, createBufferPromotionPass());
  add(mapping, createProgramMappingPass());
  add(simplification, createRangePredicatesPass());
  add(simplification, createCommonValuesPass());
}

void visitConfiguredBlock(ComponentVisitor add) {
  add({"block", "configuration", true}, createMaterializeConfigurationsPass());
  add({"block", "configured", false}, createFuseIndependentTraversalsPass());
  add({"block", "configured", false}, createNormalizeCompletedReductionsPass());
  add({"block", "configured", false}, createPeelTraversalTailsPass());
}

class RealizeSharedProgramPass
    : public impl::RealizeSharedProgramPassBase<RealizeSharedProgramPass> {
public:
  using RealizeSharedProgramPassBase::RealizeSharedProgramPassBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto transform = [&]() -> LogicalResult {
      if (failed(verifyBlockProgram(module))) return failure();
      auto kernel = getPhysicalKernel(module);
      if (failed(kernel)) return failure();
      OpPassManager pipeline(ModuleOp::getOperationName());
      visitBlockRealization([&](TransformationComponent component, std::unique_ptr<Pass> pass) {
        addTransformation(pipeline, component, std::move(pass));
      }, preferredResidentPrograms);
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
      if (failed(verifyBlockProgram(module))) return failure();
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
      if (failed(verifyBlockProgram(module))) return failure();
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
#include "Intent/Dialect/Block/Transforms/Passes.h.inc"

void registerBlockPasses() {
  registerIntentBlockTransformPasses();
  visitBlockRealization(registerTransformation, 1);
  visitConfiguredBlock(registerTransformation);
  PassPipelineRegistration<>("intent-block-shared",
      "Realize and close the shared executable Block program",
      [](OpPassManager &manager) { buildSharedBlockPipeline(manager); });
}

LogicalResult completeBlockProgramConstruction(ModuleOp module) {
  // Construction closes the initial indexed access graph. Ownership/blocking
  // and structured realization are subsequent transformations of this program.
  if (failed(realizeAccessComposition(module)))
    return failure();
  return verifyBlockProgram(module);
}

void buildSharedBlockPipeline(OpPassManager &manager,
                            unsigned preferredResidentPrograms) {
  RealizeSharedProgramPassOptions options;
  options.preferredResidentPrograms = preferredResidentPrograms;
  manager.addPass(createRealizeSharedProgramPass(options));
  visitConfiguredBlock([&](TransformationComponent component, std::unique_ptr<Pass> pass) {
    addTransformation(manager, component, std::move(pass));
  });
}

} // namespace intent::block
