#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "Legalize.h"
#include "Intent/Dialect/CPU/IR/CPUDialect.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Transforms/Configuration/FinalizedCandidates.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace intent::mojo {

#define GEN_PASS_DEF_MOJOMATERIALIZEPROGRAM
#define GEN_PASS_DEF_MOJOFUSEPRIVATECOMPUTATIONS
#define GEN_PASS_DEF_MOJOVECTORIZEPROGRAM
#define GEN_PASS_DEF_MOJOFINALIZEPROGRAM
#include "Intent/Target/Mojo/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Intent/Target/Mojo/Transforms/Passes.h.inc"

namespace {

LogicalResult verifyGroupInput(ModuleOp module, StringRef name) {
  if (failed(cpu::verifyImplementationBindings(module, "mojo")) ||
      failed(cpu::verifyCPUProgram(module, cpu::CPUProgramStage::Buffers)))
    return module.emitError() << "Mojo precondition failed: " << name;
  return success();
}

LogicalResult finishGroup(ModuleOp module, StringRef name, LogicalResult result,
                          bool realized = false) {
  if (failed(result))
    return module.emitError() << "Mojo transformation failed: " << name;
  if (failed(cpu::verifyCPUProgram(module, realized ? cpu::CPUProgramStage::Realized
                                                  : cpu::CPUProgramStage::Buffers)))
    return module.emitError() << "Mojo postcondition failed: " << name;
  return success();
}

class MaterializeProgramPass : public impl::MojoMaterializeProgramBase<MaterializeProgramPass> {
public:
  using MojoMaterializeProgramBase::MojoMaterializeProgramBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyGroupInput(module, getArgument()))) return signalPassFailure();
    if (failed(finishGroup(module, getArgument(), prepareNativeProgram(module))))
      signalPassFailure();
  }
};

class FusePrivateComputationsPass : public impl::MojoFusePrivateComputationsBase<FusePrivateComputationsPass> {
public:
  using MojoFusePrivateComputationsBase::MojoFusePrivateComputationsBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    MojoFusePrivateComputationsBase::getDependentDialects(registry);
    registry.insert<affine::AffineDialect>();
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyGroupInput(module, getArgument()))) return signalPassFailure();
    if (failed(finishGroup(module, getArgument(), fusePrivateComputations(module))))
      signalPassFailure();
  }
};

class VectorizeProgramPass : public impl::MojoVectorizeProgramBase<VectorizeProgramPass> {
public:
  using MojoVectorizeProgramBase::MojoVectorizeProgramBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyGroupInput(module, getArgument()))) return signalPassFailure();
    if (failed(finishGroup(module, getArgument(),
                          vectorizeNativeProgram(module, fuseSharedTraversals.getValue()))))
      signalPassFailure();
  }
};

class FinalizeProgramPass : public impl::MojoFinalizeProgramBase<FinalizeProgramPass> {
public:
  using MojoFinalizeProgramBase::MojoFinalizeProgramBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyGroupInput(module, getArgument()))) return signalPassFailure();
    auto result = finalizeNativeProgram(module);
    if (succeeded(result)) cpu::deduplicateFinalizedCandidates(module);
    if (failed(finishGroup(module, getArgument(), result, true))) signalPassFailure();
  }
};

} // namespace

void registerMojoPasses() {
  registerMojoTransformPasses();
  PassPipelineRegistration<>("intent-mojo-pipeline", "Legalize a configured CPU program for Mojo",
      buildMojoPipeline);
}

void buildMojoPipeline(OpPassManager &manager) {
  auto normalize = [&]() {
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass());
  };
  manager.addPass(createMojoMaterializeProgram());
  normalize();
  manager.addPass(createMojoFusePrivateComputations());
  normalize();
  manager.addPass(createMojoVectorizeProgram());
  normalize();
  manager.addPass(createMojoFinalizeProgram());
}

} // namespace intent::mojo
