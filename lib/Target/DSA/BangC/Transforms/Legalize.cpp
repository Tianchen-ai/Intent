#include "PassDetail.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "Intent/Transforms/Components.h"

using namespace mlir;
namespace intent::bangc {

#define GEN_PASS_DEF_BANGCPREPAREPROGRAM
#define GEN_PASS_DEF_BANGCNATIVECOMPUTATIONS
#define GEN_PASS_DEF_BANGCNATIVEWORKSPACE
#define GEN_PASS_DEF_BANGCFINALIZEWORKSPACE
#define GEN_PASS_DEF_BANGCNATIVEIMPLEMENTATIONS
#define GEN_PASS_DEF_BANGCLOCALCOMPOSITION
#define GEN_PASS_DEF_BANGCSTORAGEREUSE
#define GEN_PASS_DEF_BANGCSUPPLYSYNCHRONIZATION
#define GEN_PASS_DEF_BANGCSTORAGEBINDING
#include "Intent/Target/DSA/BangC/Passes.h.inc"

namespace {
LogicalResult checkProgram(ModuleOp module) {
  auto hardware = module->getAttrOfType<bangc::physical::CapabilitiesAttr>("intent_bangc.capabilities");
  if (!hardware || failed(dsa::verifyResources(hardware, [&] { return module.emitError(); })))
    return failure();
  if (usableSramBytes(module) <= 0)
    return module.emitError("BANG C requires SRAM exceeding the SDK reserve");
  if (failed(bangc::physical::verifyRealizedProgram(module))) return failure();
  auto architecture = module->getAttrOfType<StringAttr>("bangc.architecture");
  if (!architecture || architecture.getValue() != "mtp_372")
    return module.emitError("BANG C transformations require the selected mtp_372 implementation profile");
  return success();
}

OpPassManager cleanupPipeline() {
  OpPassManager manager(ModuleOp::getOperationName());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
  return manager;
}

LogicalResult finishGroup(ModuleOp module, StringRef name, LogicalResult result) {
  if (failed(result)) return module.emitError() << "BANG C transformation failed: " << name;
  if (failed(bangc::physical::verifyRealizedProgram(module))) return module.emitError() << "BANG C postcondition failed: " << name;
  return success();
}

struct PrepareProgramPass : impl::BangCPrepareProgramBase<PrepareProgramPass> {
  using Base::Base;
  void runOnOperation() final {
    auto module = getOperation();
    if (architecture.getValue() != "mtp_372") {
      module.emitError("BANG C requires an explicit mtp_372 implementation profile");
      return signalPassFailure();
    }
    if (failed(bangc::physical::verifyRealizedProgram(module))) return signalPassFailure();
    module->setAttr("bangc.architecture", StringAttr::get(module.getContext(), architecture.getValue()));
    if (failed(checkProgram(module))) return signalPassFailure();
  }
};

struct NativeComputationsPass : impl::BangCNativeComputationsBase<NativeComputationsPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(checkProgram(module))) return signalPassFailure();
    if (failed(finishGroup(module, getArgument(), realizeNativeComputations(module)))) {
      signalPassFailure();
      return;
    }
  }
};

struct NativeWorkspacePass : impl::BangCNativeWorkspaceBase<NativeWorkspacePass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(checkProgram(module))) return signalPassFailure();
    auto cleanup = cleanupPipeline();
    auto result = realizeNativeWorkspace(module, [&] { return runPipeline(cleanup, module); });
    if (failed(finishGroup(module, getArgument(), result))) {
      signalPassFailure();
      return;
    }
  }
};

struct NativeImplementationsPass : impl::BangCNativeImplementationsBase<NativeImplementationsPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(checkProgram(module))) return signalPassFailure();
    if (failed(finishGroup(module, getArgument(), selectNativeImplementations(module)))) {
      signalPassFailure();
      return;
    }
  }
};

struct FinalizeWorkspacePass : impl::BangCFinalizeWorkspaceBase<FinalizeWorkspacePass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(checkProgram(module))) return signalPassFailure();
    auto cleanup = cleanupPipeline();
    if (failed(finishGroup(module, getArgument(),
            finalizeNativeWorkspace(module, [&] { return runPipeline(cleanup, module); }))))
      signalPassFailure();
  }
};

struct LocalCompositionPass : impl::BangCLocalCompositionBase<LocalCompositionPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(checkProgram(module))) return signalPassFailure();
    auto cleanup = cleanupPipeline();
    auto result = composeLocalProgram(module, [&] { return runPipeline(cleanup, module); });
    if (failed(finishGroup(module, getArgument(), result))) {
      signalPassFailure();
      return;
    }
  }
};

struct StorageReusePass : impl::BangCStorageReuseBase<StorageReusePass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyUnboundNativeProgram(module))) return signalPassFailure();
    auto cleanup = cleanupPipeline();
    auto result = reuseStorageScopes(module, [&] { return runPipeline(cleanup, module); });
    if (failed(finishGroup(module, getArgument(), result))) return signalPassFailure();
    if (failed(verifyUnboundNativeProgram(module))) {
      module.emitError() << "BANG C native postcondition failed: " << getArgument();
      signalPassFailure();
    }
  }
};

struct SupplySynchronizationPass : impl::BangCSupplySynchronizationBase<SupplySynchronizationPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(checkProgram(module))) return signalPassFailure();
    if (failed(finishGroup(module, getArgument(), scheduleProgramSupply(module)))) {
      signalPassFailure();
      return;
    }
  }
};

struct StorageBindingPass : impl::BangCStorageBindingBase<StorageBindingPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(checkProgram(module))) return signalPassFailure();
    if (failed(finishGroup(module, getArgument(), bindProgramStorage(module)))) {
      signalPassFailure();
      return;
    }
    if (failed(verifyProgram(module))) {
      module.emitError() << "BANG C target postcondition failed: " << getArgument();
      signalPassFailure();
    }
  }
};

} // namespace

void visitBangCComponents(
    llvm::function_ref<void(TransformationComponent, std::unique_ptr<Pass>)> add,
    StringRef architecture) {
  BangCPrepareProgramOptions options;
  options.architecture = architecture.str();
  add({"bangc", "physical", true}, createBangCPrepareProgram(options));
  add({"bangc", "microkernels", true}, createBangCMatrixMicrokernels());
  add({"bangc", "microkernels", true}, createBangCNativeComputations());
  add({"bangc", "workspace", true}, createBangCNativeWorkspace());
  add({"bangc", "workspace", false}, createBangCCollectiveMicrokernels());
  add({"bangc", "workspace", true}, createBangCFinalizeWorkspace());
  add({"bangc", "native", true}, createBangCNativeImplementations());
  add({"bangc", "native", false}, createBangCLocalComposition());
  add({"bangc", "native", false}, createBangCStorageReuse());
  add({"bangc", "synchronization", true}, createBangCSupplySynchronization());
  add({"bangc", "storage", true}, createBangCStorageBinding());
}

void buildBangCPipeline(OpPassManager &manager, StringRef architecture) {
  visitBangCComponents([&](TransformationComponent component, std::unique_ptr<Pass> pass) {
    addTransformation(manager, component, std::move(pass));
  }, architecture);
}

void registerBangCPipelines() {
  visitBangCComponents(registerTransformation, "mtp_372");
  struct Options : PassPipelineOptions<Options> {
    Option<std::string> architecture{*this, "architecture",
        llvm::cl::desc("Explicit BANG C implementation architecture"), llvm::cl::init("")};
  };
  PassPipelineRegistration<Options>("intent-bangc", "Legalize a complete DSA program to BANG C",
      [](OpPassManager &manager, const Options &options) {
        buildBangCPipeline(manager, options.architecture.getValue());
      });
}

} // namespace intent::bangc
