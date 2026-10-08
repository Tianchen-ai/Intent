#include "Intent/Target/CuTile/Transforms/Passes.h"
#include "Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
namespace intent::cutile {
#define GEN_PASS_DEF_PREPAREPROGRAMPASS
#define GEN_PASS_DEF_NATIVEPROGRAMPASS
#define GEN_PASS_DEF_FINALIZEPROGRAMPASS
#include "Intent/Target/CuTile/Transforms/Passes.h.inc"

namespace {

LogicalResult finishGroup(ModuleOp module, StringRef name, LogicalResult result) {
  if (failed(result))
    return module.emitError() << "CuTile transformation failed: " << name;
  if (failed(mlir::verify(module)))
    return module.emitError() << "CuTile structural postcondition failed: " << name;
  return success();
}

class PrepareProgramPass : public impl::PrepareProgramPassBase<PrepareProgramPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), prepareProgram(module))))
      signalPassFailure();
  }
};

class NativeProgramPass : public impl::NativeProgramPassBase<NativeProgramPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), formNativeProgram(module))))
      signalPassFailure();
  }
};

class FinalizeProgramPass : public impl::FinalizeProgramPassBase<FinalizeProgramPass> {
public:
  using FinalizeProgramPassBase::FinalizeProgramPassBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(),
                          finalizeProgram(module, hoistLoopInvariants))))
      signalPassFailure();
  }
};

} // namespace

#define GEN_PASS_REGISTRATION
#include "Intent/Target/CuTile/Transforms/Passes.h.inc"

void registerCuTilePasses() {
  registerIntentCuTileTransformPasses();
  PassPipelineRegistration<>("intent-cutile-lower",
      "Legalize the complete CuTile provider program", buildCuTilePipeline);
}

void buildCuTilePipeline(OpPassManager &manager) {
  manager.addPass(createPrepareProgramPass());
  manager.addPass(createNativeProgramPass());
  manager.addPass(createFinalizeProgramPass());
}

} // namespace intent::cutile
