#include "PassDetail.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace intent::bangc {

#define GEN_PASS_DEF_BANGCPREPAREPROGRAM
#define GEN_PASS_DEF_BANGCNATIVECOMPUTATIONS
#define GEN_PASS_DEF_BANGCNATIVEWORKSPACE
#define GEN_PASS_DEF_BANGCNATIVEIMPLEMENTATIONS
#define GEN_PASS_DEF_BANGCLOCALCOMPOSITION
#define GEN_PASS_DEF_BANGCSTORAGEREUSE
#define GEN_PASS_DEF_BANGCSUPPLYSYNCHRONIZATION
#define GEN_PASS_DEF_BANGCSTORAGEBINDING
#include "Intent/Target/BangC/Passes.h.inc"

namespace {
LogicalResult checkProgram(ModuleOp module) {
  if (failed(dsa::verifyRealizedProgram(module))) return failure();
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
  if (failed(dsa::verifyRealizedProgram(module))) return module.emitError() << "BANG C postcondition failed: " << name;
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
    if (failed(dsa::verifyRealizedProgram(module))) return signalPassFailure();
    module->setAttr("bangc.architecture", StringAttr::get(module.getContext(), architecture.getValue()));
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

void buildBangCPipeline(OpPassManager &manager, StringRef architecture) {
  BangCPrepareProgramOptions options;
  options.architecture = architecture.str();
  manager.addPass(createBangCPrepareProgram(options));
  manager.addPass(createBangCNativeComputations());
  manager.addPass(createBangCNativeWorkspace());
  manager.addPass(createBangCNativeImplementations());
  manager.addPass(createBangCLocalComposition());
  manager.addPass(createBangCStorageReuse());
  manager.addPass(createBangCSupplySynchronization());
  manager.addPass(createBangCStorageBinding());
}

void registerBangCPipelines() {
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
