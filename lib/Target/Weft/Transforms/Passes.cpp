#include "Intent/Target/Weft/Transforms/Passes.h"
#include "Views.h"
#include "Intent/Target/Weft/IR/Program.h"
#include "Intent/Target/Weft/IR/WeftDialect.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Dialect/CPU/IR/CPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace intent::weft_provider {

#define GEN_PASS_DEF_WEFTPREPARETASKVIEWS
#define GEN_PASS_DEF_WEFTCONVERTPROGRAM
#define GEN_PASS_DEF_WEFTVERIFYPROGRAM
#include "Intent/Target/Weft/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Intent/Target/Weft/Transforms/Passes.h.inc"

namespace {

class PrepareTaskViewsPass : public impl::WeftPrepareTaskViewsBase<PrepareTaskViewsPass> {
public:
  using WeftPrepareTaskViewsBase::WeftPrepareTaskViewsBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(cpu::verifyCPUProgram(module, cpu::CPUProgramStage::Buffers)) ||
        failed(cpu::verifyImplementationBindings(module, "weft")))
      return signalPassFailure();
    for (auto function : module.getOps<func::FuncOp>())
      if (failed(reifyTaskViewCaptures(function))) return signalPassFailure();
    if (failed(cpu::verifyCPUProgram(module, cpu::CPUProgramStage::Buffers))) signalPassFailure();
  }
};

class ConvertProgramPass : public impl::WeftConvertProgramBase<ConvertProgramPass> {
public:
  using WeftConvertProgramBase::WeftConvertProgramBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(cpu::verifyImplementationBindings(module, "weft")) ||
        failed(legalizeProgram(module))) signalPassFailure();
  }
};

class VerifyProgramPass : public impl::WeftVerifyProgramBase<VerifyProgramPass> {
public:
  using WeftVerifyProgramBase::WeftVerifyProgramBase;
  void runOnOperation() final {
    if (failed(verifyProgram(getOperation()))) return signalPassFailure();
    markAllAnalysesPreserved();
  }
};

} // namespace

void registerWeftPasses() {
  registerWeftTransformPasses();
  PassPipelineRegistration<>("intent-weft-pipeline",
      "Lower and verify the complete Weft host and device program", buildWeftPipeline);
}

void buildWeftPipeline(OpPassManager &manager) {
  manager.addPass(createWeftPrepareTaskViews());
  manager.addPass(cpu::createCPUReuseScratchStorage());
  manager.addPass(cpu::createCPUOptimizeMemoryAccesses());
  manager.addPass(createWeftConvertProgram());
  manager.addPass(createWeftVerifyProgram());
}

} // namespace intent::weft_provider
