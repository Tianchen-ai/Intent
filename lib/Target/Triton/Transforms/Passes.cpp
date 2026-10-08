#include "Intent/Target/Triton/Transforms/Passes.h"
#include "Program.h"
#include "Intent/Target/Triton/Transforms/Mapping/ProgramGrid.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
namespace intent::triton {
#define GEN_PASS_DEF_PROGRAMGRIDPASS
#define GEN_PASS_DEF_PREPAREMEMORYPASS
#define GEN_PASS_DEF_NATIVEFORMSPASS
#define GEN_PASS_DEF_FINALIZEPROGRAMPASS
#include "Intent/Target/Triton/Transforms/Passes.h.inc"

namespace {

LogicalResult finishGroup(ModuleOp module, StringRef name, LogicalResult result) {
  if (failed(result))
    return module.emitError() << "Triton transformation failed: " << name;
  if (failed(mlir::verify(module)))
    return module.emitError() << "Triton structural postcondition failed: " << name;
  return success();
}

class ProgramGridPass : public impl::ProgramGridPassBase<ProgramGridPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(gpu::verifyGPUProgram(module)) ||
        failed(finishGroup(module, getArgument(), legalizeProgramGrid(module))) ||
        failed(gpu::verifyGPUProgram(module)))
      signalPassFailure();
  }
};

class PrepareMemoryPass : public impl::PrepareMemoryPassBase<PrepareMemoryPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), prepareTritonMemory(module))))
      signalPassFailure();
  }
};

class NativeFormsPass : public impl::NativeFormsPassBase<NativeFormsPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), formTritonProgram(module))))
      signalPassFailure();
  }
};

class FinalizeProgramPass : public impl::FinalizeProgramPassBase<FinalizeProgramPass> {
public:
  using FinalizeProgramPassBase::FinalizeProgramPassBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(),
                          finalizeTritonProgram(module, hoistLoopInvariants))))
      signalPassFailure();
  }
};

} // namespace

#define GEN_PASS_REGISTRATION
#include "Intent/Target/Triton/Transforms/Passes.h.inc"

void registerTritonPasses() {
  registerIntentTritonTransformPasses();
  PassPipelineRegistration<>("intent-triton-lower",
      "Legalize the complete Triton provider program", buildTritonPipeline);
}

void buildTritonPipeline(OpPassManager &manager) {
  manager.addPass(createProgramGridPass());
  manager.addPass(createPrepareMemoryPass());
  manager.addPass(createNativeFormsPass());
  manager.addPass(createFinalizeProgramPass());
}

} // namespace intent::triton
