#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Target/Mojo/Serialization/Serializer.h"
#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#ifdef INTENT_HAS_WEFT_CANONICAL
#include "Intent/Target/Weft/IR/WeftDialect.h"
#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "Intent/Target/Weft/Transforms/Passes.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#endif

using namespace mlir;

namespace intent::compiler {

ArrayRef<Backend> cpuBackends() {
  static const Backend adapters[] = {
      {Provider::Mojo, "mojo", true, CPUBackend{true, "mojo.json", mojo::implementations},
       nullptr, mojo::registerMojoPasses,
       [](OpPassManager &manager, const Request &) { mojo::buildMojoPipeline(manager); },
       mojo::serializeProgram},
#ifdef INTENT_HAS_WEFT_CANONICAL
      {Provider::Weft, "weft", true, CPUBackend{false, "weft.json", weft_provider::implementations},
       [](DialectRegistry &registry) {
         registry.insert<weft_provider::IntentWeftDialect, ::weft::kernel::WEFTKernelDialect>();
       },
       weft_provider::registerWeftPasses,
       [](OpPassManager &manager, const Request &) {
         weft_provider::buildWeftPipeline(manager);
       },
       weft_provider::serializeProgram}
#else
      {Provider::Weft, "weft", false, CPUBackend{false, "weft.json", nullptr},
       nullptr, nullptr, nullptr, nullptr}
#endif
  };
  return adapters;
}

void CPUBackend::buildConstruction(OpPassManager &manager, const Request &) const {
  ConstructCPUOptions options;
  options.stridedInputs = stridedInputs;
  manager.addPass(createConstructCPU(options));
}

void CPUBackend::buildShared(OpPassManager &manager, const Request &request, StringRef provider) const {
  cpu::CPUCompilationOptions options;
  options.provider = provider.str();
  llvm::SmallString<256> filename(request.profileDirectory);
  llvm::sys::path::append(filename, profileFilename);
  options.defaults = std::string(filename);
  options.overrides = request.tuningConfig;
  options.vectorBits = request.cpu.vectorBits;
  options.workers = request.cpu.workers;
  options.matrixI8I32 = request.cpu.matrixI8I32;
  options.privateBytes = request.cpu.privateBytes;
  cpu::buildCPUPipeline(manager, options);
}

LogicalResult CPUBackend::verifySharedInput(ModuleOp module, const Request &request, StringRef provider) const {
  if (failed(cpu::verifyCPUProgram(module, cpu::CPUProgramStage::Buffers))) return failure();
  auto expected = cpu::CapabilitiesAttr::getChecked([&] { return module.emitError(); },
      module.getContext(), request.cpu.vectorBits, request.cpu.workers,
      request.cpu.privateBytes, request.cpu.matrixI8I32);
  if (!expected) return failure();
  if (module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities") != expected)
    return module.emitError("shared CPU capabilities disagree with the requested CPU resources");
  auto registry = cpu::lookupImplementationProvider(module, provider);
  if (failed(registry)) return failure();
  if (failed((**registry).verifyBindings(module))) return failure();
  bool hasCandidate = false;
  for (auto function : module.getOps<func::FuncOp>()) {
    auto requirements = function->getAttrOfType<cpu::EntryRequirementsAttr>(cpu::entryRequirementsAttr);
    auto summary = function->getAttrOfType<ArrayAttr>("intent_cpu.implementations");
    if (function.isExternal() || !summary || summary.empty() ||
        !function->getAttrOfType<cpu::ConfigurationAttr>("intent_cpu.configuration") ||
        !function->getAttrOfType<BoolAttr>("intent_cpu.requires_matrix_i8_i32"))
      return function.emitError("shared CPU input requires executable candidates with complete bound configuration and implementation summaries");
    if (requirements.getContiguousViews() != !stridedInputs)
      return function.emitError("shared CPU entry layout disagrees with the selected provider");
    hasCandidate = true;
    auto status = function.walk([&](Operation *operation) -> WalkResult {
      if (!operation->hasAttr("intent_cpu.implementation")) return WalkResult::advance();
      return succeeded((**registry).lookup(operation)) ? WalkResult::advance() : WalkResult::interrupt();
    });
    if (status.wasInterrupted()) return failure();
  }
  if (!hasCandidate) return module.emitError("shared CPU input requires at least one executable candidate");
  return success();
}

} // namespace intent::compiler
