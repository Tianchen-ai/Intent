#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Compiler/TargetOptions.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"

using namespace mlir;

namespace intent::compiler {

LogicalResult configureCPUTarget(ModuleOp module, Request &request) {
  auto options = readTargetOptions(module, request,
      {"vector_bits", "workers", "private_bytes", "matrix_i8_i32", "matrix_bf16_f32"});
  if (failed(options)) return failure();
  auto &target = request.cpu;
  if (failed(readIntegerOption(module, *options, "vector_bits", target.vectorBits)) ||
      failed(readIntegerOption(module, *options, "workers", target.workers)) ||
      failed(readIntegerOption(module, *options, "private_bytes", target.privateBytes)) ||
      failed(readBooleanOption(module, *options, "matrix_i8_i32", target.matrixI8I32)) ||
      failed(readBooleanOption(module, *options, "matrix_bf16_f32", target.matrixBF16F32)))
    return failure();
  return success();
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
  options.matrixBF16F32 = request.cpu.matrixBF16F32;
  options.privateBytes = request.cpu.privateBytes;
  cpu::buildCPUPipeline(manager, options);
}

LogicalResult CPUBackend::verifySharedInput(ModuleOp module, const Request &request, StringRef provider) const {
  if (failed(cpu::verifyCPUProgram(module, cpu::CPUProgramStage::Buffers))) return failure();
  auto expected = cpu::CapabilitiesAttr::getChecked([&] { return module.emitError(); },
      module.getContext(), request.cpu.vectorBits, request.cpu.workers,
      request.cpu.privateBytes, request.cpu.matrixI8I32, request.cpu.matrixBF16F32);
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
