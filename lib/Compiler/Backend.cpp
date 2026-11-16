#include "Intent/Compiler/Backend.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

using namespace mlir;

namespace intent::compiler {

#define INTENT_BACKEND(Name) Backend create##Name##Backend();
#include "Intent/Compiler/Backends.inc"
#undef INTENT_BACKEND

ArrayRef<Backend> backends() {
  static const SmallVector<Backend> catalog = [] {
    SmallVector<Backend> result;
#define INTENT_BACKEND(Name) result.push_back(create##Name##Backend());
#include "Intent/Compiler/Backends.inc"
#undef INTENT_BACKEND
    return result;
  }();
  return catalog;
}

const Backend &backend(Provider provider) {
  for (const Backend &adapter : backends())
    if (adapter.provider == provider) return adapter;
  llvm_unreachable("invalid provider");
}

SmallVector<std::string> Backend::profilePaths(StringRef directory) const {
  SmallVector<std::string> result;
  if (!available)
    return result;
  for (StringRef filename : model->profileFiles()) {
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, filename);
    result.push_back(std::string(path));
  }
  return result;
}

void Backend::buildConstruction(OpPassManager &manager, const Request &request) const {
  model->buildConstruction(manager, request);
}

void Backend::buildShared(OpPassManager &manager, const Request &request) const {
  model->buildShared(manager, request, name);
}

LogicalResult Backend::verifySharedInput(ModuleOp module, const Request &request) const {
  return model->verifySharedInput(module, request, name);
}

SmallVector<block::TuningProfileSource> blockProfileSources(StringRef directory,
                                                     StringRef provider) {
  SmallVector<block::TuningProfileSource> sources;
  auto append = [&](block::TuningProfileSchema schema, StringRef resource) {
    llvm::SmallString<256> filename(directory);
    llvm::sys::path::append(filename, resource);
    sources.push_back({schema, std::string(filename)});
  };
  for (const Backend &adapter : backends())
    if (adapter.available && adapter.name == provider)
      if (auto schema = adapter.model->blockProfiles())
        for (StringRef filename : adapter.model->profileFiles())
          append(*schema, filename);
  return sources;
}

} // namespace intent::compiler
