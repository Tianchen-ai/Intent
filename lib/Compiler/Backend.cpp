#include "Intent/Compiler/Backend.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

using namespace mlir;

namespace intent::compiler {

// Each family declares its compiled adapters next to its construction policy.
llvm::ArrayRef<Backend> gpuBackends();
llvm::ArrayRef<Backend> cpuBackends();
llvm::ArrayRef<Backend> dsaBackends();

ArrayRef<Backend> backends() {
  static const SmallVector<Backend> catalog = [] {
    SmallVector<Backend> result;
    llvm::append_range(result, gpuBackends());
    llvm::append_range(result, cpuBackends());
    llvm::append_range(result, dsaBackends());
    return result;
  }();
  return catalog;
}

const Backend &backend(Provider provider) {
  for (const Backend &adapter : backends())
    if (adapter.provider == provider) return adapter;
  llvm_unreachable("invalid provider");
}

Family Backend::family() const {
  return std::visit([](const auto &model) { return model.family; }, model);
}

SmallVector<std::string> Backend::profilePaths(StringRef directory) const {
  SmallVector<std::string> result;
  if (!available)
    return result;
  if (std::holds_alternative<GPUBackend>(model)) {
    for (const auto &source : gpuProfileSources(directory, name))
      result.push_back(source.filename);
  } else if (const auto *cpu = std::get_if<CPUBackend>(&model)) {
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, cpu->profileFilename);
    result.push_back(std::string(path));
  } else if (std::holds_alternative<DSABackend>(model)) {
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, "bangc.json");
    result.push_back(std::string(path));
  }
  return result;
}

void Backend::buildConstruction(OpPassManager &manager, const Request &request) const {
  std::visit([&](const auto &model) { model.buildConstruction(manager, request); }, model);
}

void Backend::buildShared(OpPassManager &manager, const Request &request) const {
  std::visit([&](const auto &model) { model.buildShared(manager, request, name); }, model);
}

LogicalResult Backend::verifySharedInput(ModuleOp module, const Request &request) const {
  return std::visit([&](const auto &model) {
    return model.verifySharedInput(module, request, name);
  }, model);
}

SmallVector<gpu::TuningProfileSource> gpuProfileSources(StringRef directory,
                                                     StringRef provider) {
  SmallVector<gpu::TuningProfileSource> sources;
  auto append = [&](gpu::TuningProfileSchema schema, StringRef resource) {
    llvm::SmallString<256> filename(directory);
    llvm::sys::path::append(filename, resource);
    sources.push_back({schema, std::string(filename)});
  };
  for (const Backend &adapter : backends())
    if (adapter.available && adapter.name == provider)
      if (const auto *model = std::get_if<GPUBackend>(&adapter.model))
        append(model->profiles, model->profileFilename);
  return sources;
}

} // namespace intent::compiler
