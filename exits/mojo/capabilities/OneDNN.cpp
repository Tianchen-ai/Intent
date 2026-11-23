#include "compilerinfra/Assembly.h"
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <sched.h>
#include <sstream>
#include <string>
#include <vector>
#ifdef CI_ONEDNN_HEADER_AVAILABLE
#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_ukernel.h"
#endif

namespace CI_UNIT_NAMESPACE {
namespace {
bool cpuSetHasAMXInt8() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
    return false;
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  int proc = -1;
  bool seen = false;
  while (std::getline(in, line)) {
    auto pos = line.find(':');
    if (pos == std::string::npos)
      continue;
    auto key = llvm::StringRef(line.data(), pos).trim();
    auto val = llvm::StringRef(line.data() + pos + 1).trim();
    if (key == "processor") {
      proc = std::stoi(val.str());
    } else if (key == "flags" && proc >= 0 && CPU_ISSET(proc, &mask)) {
      seen = true;
      bool tile = false, int8 = false;
      llvm::SmallVector<llvm::StringRef> flags;
      val.split(flags, ' ', -1, false);
      for (auto f : flags) {
        tile |= (f == "amx_tile");
        int8 |= (f == "amx_int8");
      }
      if (!tile || !int8)
        return false;
    }
  }
  return seen;
}
} // namespace

llvm::json::Object probeOneDNN() {
  llvm::json::Object out;
#ifndef CI_ONEDNN_HEADER_AVAILABLE
  out["available"] = false;
  out["reason"] = "built without oneDNN >=3.13,<4 headers";
  return out;
#elif !defined(DNNL_EXPERIMENTAL_UKERNEL)
  out["available"] = false;
  out["reason"] = "oneDNN headers built without ONEDNN_EXPERIMENTAL_UKERNEL";
  return out;
#else
  const char *overrideLib = std::getenv("INTENT_ONEDNN_LIB");
  std::string candidate = (overrideLib && *overrideLib)
                              ? std::string(overrideLib)
                              : std::string(CI_ONEDNN_LIBRARY_PATH);
  void *handle = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle && !(overrideLib && *overrideLib))
    handle = dlopen("libdnnl.so.3", RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    out["available"] = false;
    const char *err = dlerror();
    out["reason"] = std::string("dlopen failed: ") + (err ? err : candidate);
    return out;
  }
  struct Guard {
    void *h;
    ~Guard() {
      if (h)
        dlclose(h);
    }
  } guard{handle};

  using VersionFn = const dnnl_version_t *(*)();
  using CreateFn = dnnl_status_t (*)(
      dnnl_brgemm_t *, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t,
      dnnl_dim_t, dnnl_dim_t, dnnl_dim_t, dnnl_data_type_t, dnnl_data_type_t,
      dnnl_data_type_t);
  using GetPackFn = dnnl_status_t (*)(dnnl_pack_type_t *, dnnl_data_type_t,
                                      dnnl_data_type_t);
  using StatusFn = dnnl_status_t (*)(dnnl_brgemm_t);
  using ScratchFn = dnnl_status_t (*)(const_dnnl_brgemm_t, size_t *);
  using CtxSetFn = dnnl_status_t (*)(const_dnnl_brgemm_t);
  using CtxRelFn = dnnl_status_t (*)();
  using ExecFn = dnnl_status_t (*)(const_dnnl_brgemm_t, const void *,
                                   const void *, const dnnl_dim_t *, void *,
                                   void *);

  auto sym = [&](const char *name) -> void * { return dlsym(handle, name); };
  auto fnVer = reinterpret_cast<VersionFn>(sym("dnnl_version"));
  auto fnCreate = reinterpret_cast<CreateFn>(sym("dnnl_brgemm_create"));
  auto fnGetPack =
      reinterpret_cast<GetPackFn>(sym("dnnl_brgemm_get_B_pack_type"));
  auto fnFinalize = reinterpret_cast<StatusFn>(sym("dnnl_brgemm_finalize"));
  auto fnScratch =
      reinterpret_cast<ScratchFn>(sym("dnnl_brgemm_get_scratchpad_size"));
  auto fnGenerate = reinterpret_cast<StatusFn>(sym("dnnl_brgemm_generate"));
  auto fnCtxSet = reinterpret_cast<CtxSetFn>(sym("dnnl_brgemm_set_hw_context"));
  auto fnCtxRel =
      reinterpret_cast<CtxRelFn>(sym("dnnl_brgemm_release_hw_context"));
  auto fnExec = reinterpret_cast<ExecFn>(sym("dnnl_brgemm_execute"));
  auto fnDestroy = reinterpret_cast<StatusFn>(sym("dnnl_brgemm_destroy"));

  if (fnVer) {
    if (const dnnl_version_t *v = fnVer()) {
      out["version"] = std::to_string(v->major) + "." +
                       std::to_string(v->minor) + "." +
                       std::to_string(v->patch);
    }
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(fnVer), &info) && info.dli_fname)
      out["library"] = std::string(info.dli_fname);
  }
  if (!fnVer || !fnCreate || !fnGetPack || !fnFinalize || !fnScratch ||
      !fnGenerate || !fnCtxSet || !fnCtxRel || !fnExec || !fnDestroy) {
    out["available"] = false;
    out["reason"] =
        "oneDNN runtime lacks experimental BRGeMM ukernel symbols";
    return out;
  }

  dnnl_brgemm_t brg = nullptr;
  if (fnCreate(&brg, 2, 16, 16, 1, 16, 16, 16, dnnl_f32, dnnl_f32, dnnl_f32) !=
          dnnl_success ||
      fnFinalize(brg) != dnnl_success || fnGenerate(brg) != dnnl_success) {
    if (brg)
      fnDestroy(brg);
    out["available"] = false;
    out["reason"] = "host CPU ISA cannot generate oneDNN f32 BRGeMM ukernel";
    return out;
  }
  size_t scratchSize = 0;
  fnScratch(brg, &scratchSize);
  std::vector<uint8_t> scratch(scratchSize ? scratchSize : 64, 0);
  std::vector<float> a(32, 1.0f), b(256, 0.625f), c(32, 0.0f);
  const dnnl_dim_t offs[2] = {0, 0};
  bool ok = fnCtxSet(brg) == dnnl_success &&
            fnExec(brg, a.data(), b.data(), offs, c.data(), scratch.data()) ==
                dnnl_success;
  fnCtxRel();
  fnDestroy(brg);
  if (!ok || c[0] != 10.0f || c[31] != 10.0f) {
    out["available"] = false;
    out["reason"] = "oneDNN f32 BRGeMM probe execution failed verification";
    return out;
  }

  bool amxVerified = false;
  if (cpuSetHasAMXInt8()) {
    dnnl_brgemm_t brgS8 = nullptr;
    if (fnCreate(&brgS8, 2, 16, 16, 1, 16, 16, 16, dnnl_s8, dnnl_s8,
                 dnnl_s32) == dnnl_success &&
        fnFinalize(brgS8) == dnnl_success &&
        fnGenerate(brgS8) == dnnl_success) {
      size_t s8Scratch = 0;
      fnScratch(brgS8, &s8Scratch);
      std::vector<uint8_t> ws(s8Scratch ? s8Scratch : 64, 0);
      std::vector<int8_t> as8(32, 2), bs8(256, -3);
      std::vector<int32_t> cs32(32, 0);
      if (fnCtxSet(brgS8) == dnnl_success) {
        if (fnExec(brgS8, as8.data(), bs8.data(), offs, cs32.data(),
                   ws.data()) == dnnl_success &&
            cs32[0] == -96 && cs32[31] == -96)
          amxVerified = true;
        fnCtxRel();
      }
    }
    if (brgS8)
      fnDestroy(brgS8);
  }
  out["amx_int8_verified"] = amxVerified;
  out["available"] = true;
  return out;
#endif
}

void registerOneDNN(compilerinfra::Unit &u) {
  u.dependencyProbes["onednn-brgemm"] = probeOneDNN;
}
} // namespace CI_UNIT_NAMESPACE
