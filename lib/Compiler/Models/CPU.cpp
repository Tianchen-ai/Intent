#include "Intent/Compiler/Models/CPU.h"
#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Compiler/TargetOptions.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "mlir/AsmParser/AsmParser.h"

using namespace mlir;

namespace intent::compiler {

static LogicalResult configureCPU(ModuleOp module, Request &request,
                                  ArrayRef<StringRef> optionalFields) {
  module.getContext()->getOrLoadDialect<cpu::IntentCPUDialect>();
  auto options = readTargetOptions(module, request,
      {"vector", "contractions", "execution"}, optionalFields);
  if (failed(options)) return failure();
  if (options->get("strided_inputs")) {
    auto strided = options->getArray("strided_inputs");
    if (!strided || strided->empty())
      return module.emitError("strided_inputs requires the names of the strided input views");
    Builder builder(module.getContext());
    SmallVector<Attribute> names;
    for (auto &item : *strided) {
      auto name = item.getAsString();
      if (!name || name->empty() || llvm::is_contained(names, builder.getStringAttr(*name)))
        return module.emitError("strided_inputs requires distinct nonempty parameter names");
      names.push_back(builder.getStringAttr(*name));
    }
    request.bindings = builder.getDictionaryAttr(
        {builder.getNamedAttr("strided_inputs", builder.getArrayAttr(names))});
  }
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

LogicalResult configureCPUTarget(ModuleOp module, Request &request) {
  return configureCPU(module, request, {});
}

LogicalResult configureCPUWithLayoutBinding(ModuleOp module, Request &request) {
  return configureCPU(module, request, {"strided_inputs"});
}

void CPUBackend::buildConstruction(OpPassManager &manager, const Request &request) const {
  ConstructCPUOptions options;
  options.stridedInputs = stridedInputs;
  if (auto names = request.bindings ? request.bindings.getAs<ArrayAttr>("strided_inputs") : ArrayAttr{})
    for (auto name : names) options.stridedParameters.push_back(cast<StringAttr>(name).getValue().str());
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
    auto names = request.bindings ? request.bindings.getAs<ArrayAttr>("strided_inputs") : ArrayAttr{};
    if (!requirements || requirements.getContiguousViews() != (!stridedInputs && !names))
      return function.emitError("shared CPU entry layout disagrees with the selected provider");
    if (names) {
      auto interface = getPublicInterface(function);
      size_t matched = 0;
      for (auto [index, field] : llvm::enumerate(interface.getArguments())) {
        auto view = getPublicView(interface, index);
        auto parameter = cast<PublicParameterAttr>(field);
        bool expected = llvm::is_contained(names, parameter.getName());
        bool strided = view && view.getAccess() == 0 &&
            isa<StridedLayoutAttr>(cast<MemRefType>(function.getArgument(index).getType()).getLayout());
        if (strided != expected)
          return function.emitError("shared CPU parameter layout differs from its requested binding");
        matched += expected;
      }
      if (matched != names.size())
        return function.emitError("requested strided parameter is missing from the shared CPU entry");
    }
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
