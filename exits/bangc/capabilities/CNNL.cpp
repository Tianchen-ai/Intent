#include "compilerinfra/Assembly.h"
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace CI_UNIT_NAMESPACE {
namespace {
llvm::json::Object probeCNNL() {
  llvm::json::Object out;
  std::vector<std::string> candidates;
  if (const char *overrideLib = std::getenv("INTENT_CNNL_LIB");
      overrideLib && *overrideLib) {
    candidates.emplace_back(overrideLib);
  } else {
    if (const char *neuware = std::getenv("NEUWARE_HOME");
        neuware && *neuware) {
      candidates.push_back(std::string(neuware) + "/lib64/libcnnl.so");
      candidates.push_back(std::string(neuware) + "/lib/libcnnl.so");
    }
    candidates.emplace_back("/usr/local/neuware/lib64/libcnnl.so");
    candidates.emplace_back("libcnnl.so");
  }

  void *handle = nullptr;
  std::string resolvedPath;
  std::string lastReason;
  for (const auto &cand : candidates) {
    if (cand.empty())
      continue;
    void *h = dlopen(cand.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!h) {
      if (lastReason.empty()) {
        const char *err = dlerror();
        lastReason =
            std::string("dlopen failed: ") + (err ? err : cand.c_str());
      }
      continue;
    }
    handle = h;
    resolvedPath = cand;
    break;
  }

  if (!handle) {
    out["available"] = false;
    out["reason"] =
        lastReason.empty() ? "libcnnl.so not found" : std::move(lastReason);
    return out;
  }

  struct DlGuard {
    void *h;
    ~DlGuard() {
      if (h)
        dlclose(h);
    }
  } guard{handle};

  static const char *const kRequiredSymbols[] = {
      "cnnlGetLibVersion",
      "cnnlCreate",
      "cnnlDestroy",
      "cnnlSetQueue",
      "cnnlCreateTensorDescriptor",
      "cnnlSetTensorDescriptor",
      "cnnlSetTensorDescriptor_v2",
      "cnnlSetTensorDescriptorOnchipDataType",
      "cnnlDestroyTensorDescriptor",
      "cnnlMatMulDescCreate",
      "cnnlSetMatMulDescAttr",
      "cnnlMatMulDescDestroy",
      "cnnlMatMulAlgoCreate",
      "cnnlMatMulAlgoDestroy",
      "cnnlCreateMatMulHeuristicResult",
      "cnnlDestroyMatMulHeuristicResult",
      "cnnlGetMatMulAlgoHeuristic",
      "cnnlGetMatMulHeuristicResult",
      "cnnlMatMul_v2",
      "cnnlStrideBatchMatMul",
      "cnnlSoftmaxForward_v2",
      "cnnlLayerNormForward",
      "cnnlGetLayerNormOpWorkspaceSize",
      "cnnlCreateTransposeDescriptor",
      "cnnlSetTransposeDescriptor",
      "cnnlDestroyTransposeDescriptor",
      "cnnlGetTransposeWorkspaceSize",
      "cnnlTranspose_v2",
  };

  llvm::json::Array symbols;
  for (const char *sym : kRequiredSymbols) {
    if (!dlsym(handle, sym)) {
      out["available"] = false;
      out["reason"] = std::string("missing CNNL symbol: ") + sym;
      return out;
    }
    symbols.push_back(sym);
  }

  using GetVerFn = int (*)(int *, int *, int *);
  auto getVer = reinterpret_cast<GetVerFn>(dlsym(handle, "cnnlGetLibVersion"));
  int major = 0, minor = 0, patch = 0;
  if (getVer(&major, &minor, &patch) != 0 || major <= 0) {
    out["available"] = false;
    out["reason"] = "cnnlGetLibVersion failed";
    return out;
  }

  out["available"] = true;
  out["version"] = std::to_string(major) + "." + std::to_string(minor) + "." +
                   std::to_string(patch);
  out["library"] = resolvedPath;
  out["symbols"] = std::move(symbols);
  return out;
}
} // namespace

void registerCNNL(compilerinfra::Unit &u) {
  u.dependencyProbes["cnnl-library"] = probeCNNL;
}
} // namespace CI_UNIT_NAMESPACE
