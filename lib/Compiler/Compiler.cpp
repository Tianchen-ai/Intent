#include "Intent/Compiler/Compiler.h"
#include "Intent/Compiler/Backend.h"
#include "Intent/Transforms/Passes.h"
#include "Intent/Transforms/PassManager.h"

using namespace mlir;

namespace intent::compiler {
namespace {
struct FailureDescription { Failure code; StringRef name; };
const FailureDescription failures[] = {
    {Failure::Invocation, "compiler_invocation"}, {Failure::KernelIR, "kernel_ir"},
    {Failure::Construction, "physical_program"}, {Failure::SharedPipeline, "shared_pipeline"},
    {Failure::UnavailableProvider, "provider_program"}, {Failure::ProviderPipeline, "provider_pipeline"},
    {Failure::Translation, "terminal_translation"}, {Failure::Output, "compiler_output"}};

} // namespace

std::optional<Provider> parseProvider(StringRef name) {
  for (const Backend &adapter : backends()) if (adapter.name == name) return adapter.provider;
  return std::nullopt;
}
StringRef providerName(Provider kind) {
  return backend(kind).name;
}
bool isProviderAvailable(Provider kind) {
  for (const Backend &adapter : backends())
    if (adapter.provider == kind) return adapter.available;
  return false;
}
StringRef stageName(Stage stage) {
  switch (stage) {
  case Stage::Kernel: return "kir";
  case Stage::Shared: return "shared";
  case Stage::Provider: return "provider";
  }
  llvm_unreachable("invalid compiler stage");
}
StringRef failureStage(Failure code) {
  for (const auto &failure : failures) if (failure.code == code) return failure.name;
  llvm_unreachable("not a compiler failure");
}
llvm::json::Object information(StringRef profileDirectory) {
  llvm::json::Object providers;
  for (const Backend &adapter : backends()) {
    StringRef family;
    switch (adapter.family()) {
    case Family::GPU: family = "gpu"; break;
    case Family::CPU: family = "cpu"; break;
    case Family::DSA: family = "dsa"; break;
    }
    llvm::json::Array profiles;
    for (const std::string &path : adapter.profilePaths(profileDirectory))
      profiles.push_back(path);
    providers[adapter.name] = llvm::json::Object{
        {"family", family}, {"available", adapter.available},
        {"profiles", std::move(profiles)}};
  }
  llvm::json::Object failureStages;
  for (const auto &failure : failures)
    failureStages[std::to_string(static_cast<int>(failure.code))] = failure.name;
  return llvm::json::Object{{"providers", std::move(providers)},
          {"input_stages", llvm::json::Array{"kir", "shared"}},
          {"stages", llvm::json::Array{stageName(Stage::Kernel), stageName(Stage::Shared), stageName(Stage::Provider)}},
          {"outputs", llvm::json::Object{{"kir", llvm::json::Array{"ir"}},
              {"shared", llvm::json::Array{"ir"}},
              {"provider", llvm::json::Array{"ir", "source", "metadata"}}}},
          {"compile_options", llvm::json::Object{
              {"numerics", llvm::json::Array{"source", "relaxed_normalization"}},
              {"online_reduction", true}, {"optimization_remarks", false}}},
          {"failure_stages", std::move(failureStages)}};
}

Result compile(OwningOpRef<ModuleOp> module, const Request &request) {
  Result result;
  result.module = std::move(module);
  bool sharedInput = request.inputStage == InputStage::Shared;
  result.stage = sharedInput ? Stage::Shared : Stage::Kernel;
  if (!result.module) {
    result.failure = sharedInput ? Failure::SharedPipeline : Failure::KernelIR;
    return result;
  }
  if (request.stopAfter == Stage::Kernel &&
      (request.provider || !request.tuningConfig.empty())) {
    result.module->emitError("KIR output does not select a physical target or tuning profile");
    result.failure = Failure::Invocation;
    return result;
  }
  auto existing = (*result.module)->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr);
  if ((*result.module)->hasAttr(compileOptionsAttr) && !existing) {
    result.module->emitError("intent.compile_options must use its typed policy attribute");
    result.failure = Failure::Invocation;
    return result;
  }
  auto selected = existing;
  if (request.options || (!sharedInput && !existing)) {
    CompilationOptions options = request.options.value_or(CompilationOptions{});
    selected = CompileOptionsAttr::get(result.module->getContext(), options.numerics,
                                      options.onlineReduction, options.optimizationRemarks);
  }
  if (!selected || (sharedInput && request.options && selected != existing) ||
      (existing && request.options && selected != existing)) {
    result.module->emitError("compilation options must agree with the policy already bound in the input IR");
    result.failure = Failure::Invocation;
    return result;
  }
  (*result.module)->setAttr(compileOptionsAttr, selected);
  if (sharedInput && (request.stopAfter == Stage::Kernel || !request.tuningConfig.empty())) {
    result.module->emitError("shared input requires shared/provider output and cannot reload tuning profiles");
    result.failure = Failure::Invocation;
    return result;
  }
  if (request.stopAfter != Stage::Kernel && !request.provider) {
    result.module->emitError("compilation beyond KIR requires an explicit provider");
    result.failure = Failure::Invocation;
    return result;
  }
  if (request.stopAfter != Stage::Kernel && !isProviderAvailable(*request.provider)) {
    result.module->emitError() << "provider is not compiled into this compiler: " << providerName(*request.provider);
    result.failure = Failure::UnavailableProvider;
    return result;
  }
  PassManager manager(result.module->getContext());
  if (failed(configurePassManager(manager))) { result.failure = Failure::Invocation; return result; }
  auto run = [&](Failure failure) {
    if (failed(manager.run(*result.module))) { result.failure = failure; return false; }
    manager.clear();
    return true;
  };
  if (!sharedInput) {
    manager.addPass(createNormalizeKernelIRPass());
    if (!run(Failure::KernelIR) || request.stopAfter == Stage::Kernel) return result;
  }

  result.stage = Stage::Shared;
  const Backend &adapter = backend(*request.provider);
  if (sharedInput) {
    if (failed(adapter.verifySharedInput(*result.module, request))) {
      result.failure = Failure::SharedPipeline;
      return result;
    }
  } else {
    adapter.buildConstruction(manager, request);
    if (!run(Failure::Construction)) return result;
    adapter.buildShared(manager, request);
    if (!run(Failure::SharedPipeline)) return result;
  }
  if (request.stopAfter == Stage::Shared) return result;

  result.stage = Stage::Provider;
  adapter.buildPipeline(manager, request);
  if (!run(Failure::ProviderPipeline)) return result;
  if (failed(adapter.serialize(*result.module, result.source, result.metadata)))
    result.failure = Failure::Translation;
  return result;
}
} // namespace intent::compiler
