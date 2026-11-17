#include "Intent/Compiler/Models/CPU.h"
#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Compiler/TargetOptions.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "mlir/AsmParser/AsmParser.h"

using namespace mlir;

namespace intent::compiler {

LogicalResult configureCPUTarget(ModuleOp module, Request &request) {
  module.getContext()->getOrLoadDialect<cpu::IntentCPUDialect>();
  auto options = readTargetOptions(module, request,
      {"vector", "contractions", "execution"});
  if (failed(options)) return failure();
  auto vector = options->getObject("vector"), execution = options->getObject("execution");
  auto contractions = options->getArray("contractions");
  if (!vector || vector->size() != 2 || !execution || execution->size() != 2 || !contractions)
    return module.emitError("CPU target requires vector facts, a contraction catalog and separate execution budgets");
  int64_t vectorBits, workers, privateBytes;
  bool scalable;
  if (failed(readIntegerOption(module, *vector, "minimum_bits", vectorBits)) ||
      failed(readBooleanOption(module, *vector, "scalable", scalable)) ||
      failed(readIntegerOption(module, *execution, "workers", workers)) ||
      failed(readIntegerOption(module, *execution, "private_bytes", privateBytes)))
    return failure();
  SmallVector<Attribute> catalog;
  for (auto &item : *contractions) {
    auto value = item.getAsObject();
    if (!value || value->size() != 4 || !value->getString("lhs") ||
        !value->getString("rhs") || !value->getString("accumulator") || !value->getBoolean("flush_subnormals"))
      return module.emitError("CPU contraction capability requires exact lhs/rhs/accumulator types and subnormal behavior");
    auto lhs = parseType(*value->getString("lhs"), module.getContext());
    auto rhs = parseType(*value->getString("rhs"), module.getContext());
    auto accumulator = parseType(*value->getString("accumulator"), module.getContext());
    if (!lhs || !rhs || !accumulator) return failure();
    auto capability = cpu::ContractionCapabilityAttr::getChecked([&] { return module.emitError(); },
        module.getContext(), lhs, rhs, accumulator, *value->getBoolean("flush_subnormals"));
    if (!capability) return failure();
    catalog.push_back(capability);
  }
  auto vectorAttr = cpu::VectorCapabilitiesAttr::getChecked([&] { return module.emitError(); },
      module.getContext(), vectorBits, scalable);
  auto executionAttr = cpu::ExecutionConfigAttr::getChecked([&] { return module.emitError(); },
      module.getContext(), workers, privateBytes);
  if (!vectorAttr || !executionAttr) return failure();
  request.target = cpu::CapabilitiesAttr::getChecked([&] { return module.emitError(); },
      module.getContext(), vectorAttr, ArrayAttr::get(module.getContext(), catalog), executionAttr);
  return success(static_cast<bool>(request.target));
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
  llvm::sys::path::append(filename, files[0]);
  options.defaults = std::string(filename);
  options.overrides = request.tuningConfig;
  std::string target;
  llvm::raw_string_ostream stream(target);
  request.target.print(stream);
  options.target = target;
  cpu::buildCPUValuePipeline(manager, options);
  if (pipeline) pipeline(manager, options);
}

LogicalResult CPUBackend::verifySharedInput(ModuleOp module, const Request &request, StringRef provider) const {
  if (failed(cpu::verifyCPUProgram(module, inputStage))) return failure();
  auto expected = cast<cpu::CapabilitiesAttr>(request.target);
  if (module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities") != expected)
    return module.emitError("shared CPU capabilities disagree with the requested CPU resources");
  for (auto function : module.getOps<func::FuncOp>()) {
    auto requirements = function->getAttrOfType<cpu::EntryRequirementsAttr>(cpu::entryRequirementsAttr);
    if (!requirements || requirements.getContiguousViews() != !stridedInputs)
      return function.emitError("shared CPU entry layout disagrees with the selected provider");
  }
  auto registry = cpu::lookupImplementationProvider(module, provider);
  if (failed(registry)) return failure();
  if (inputStage == cpu::CPUProgramStage::Values)
    return cpu::verifyCPUProfiles(module, **registry, provider);
  if (failed((**registry).verifyBindings(module))) return failure();
  bool hasCandidate = false;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function.isExternal() || !function->getAttrOfType<cpu::ConfigurationAttr>("intent_cpu.configuration"))
      return function.emitError("shared CPU input requires executable candidates with bound configuration");
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
