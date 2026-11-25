#include "Target.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "llvm/Support/MathExtras.h"
#include <dlfcn.h>

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
namespace {
std::string probeCudaCalibration() {
  static const std::string detected = []() -> std::string {
    void *lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib)
      return "";
    using InitFn = int (*)(unsigned);
    using GetDevFn = int (*)(int *, int);
    using GetAttrFn = int (*)(int *, int, int);
    auto cuInit = reinterpret_cast<InitFn>(dlsym(lib, "cuInit"));
    auto cuDeviceGet = reinterpret_cast<GetDevFn>(dlsym(lib, "cuDeviceGet"));
    auto cuDeviceGetAttribute =
        reinterpret_cast<GetAttrFn>(dlsym(lib, "cuDeviceGetAttribute"));
    int dev = 0, major = 0, minor = 0;
    if (cuInit && cuDeviceGet && cuDeviceGetAttribute && cuInit(0) == 0 &&
        cuDeviceGet(&dev, 0) == 0 &&
        cuDeviceGetAttribute(&major, 75, dev) == 0 &&
        cuDeviceGetAttribute(&minor, 76, dev) == 0) {
      int sm = major * 10 + minor;
      if (sm == 90)
        return "sm90";
      if (sm == 120)
        return "sm120";
    }
    return "";
  }();
  return detected;
}
} // namespace
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &options) {
  TargetProfile profile;
  auto arch = options.getInteger("sm");
  if (!arch) {
    profile.services.calibrationProfile =
        options.getString("calibration").value_or(probeCudaCalibration()).str();
    return profile; // Static layout has no bound call arguments.
  }
  if (*arch != 90 && *arch != 120)
    return llvm::createStringError("unsupported Triton GPU architecture");
  auto target = std::make_shared<intentdsl::itsrc::TritonTarget>();
  target->permitTma = options.getBoolean("allow_tma").value_or(false);
  target->scaledMma = (*arch == 90 || *arch == 120);
  if (auto err =
          profile.services.provide<intentdsl::itsrc::TritonTarget>(target))
    return std::move(err);
  auto caps = std::make_shared<intentdsl::RowCapacity>();
  caps->globalAtomicBins=true;
  auto matrix=std::make_shared<intentdsl::block::MatrixTarget>();
  matrix->estimate="exits/triton::matrix-issues";
  matrix->implementations = {
      {"contract", {"exits/triton::native-dot"}},
      {"window_contract", {"exits/triton::native-dot"}},
      {"scaled_contract",
       {"exits/triton::scaled-mma", "exits/triton::native-dot",
        "exits/triton::row-collective"}},
      {"q4k_dot", {"exits/triton::row-collective"}}};
  if (*arch == 90)
    matrix->tiles = {
        {2, 32, 256, R"({"num_warps":2,"num_stages":2})", "q4k_dot"},
        {4, 32, 256, R"({"num_warps":4,"num_stages":2})", "q4k_dot"},
        {8, 32, 256, R"({"num_warps":4,"num_stages":2})", "q4k_dot"},
        {16, 32, 32, R"({"num_warps":4,"num_stages":2})"},
        {32, 64, 32, R"({"num_warps":4,"num_stages":3})"},
        {32, 64, 64, R"({"num_warps":4,"num_stages":3})", "contract"},
        {64, 64, 32, R"({"num_warps":4,"num_stages":3})"},
        {64, 128, 64, R"({"num_warps":8,"num_stages":3})"},
        {32, 128, 128, R"({"num_warps":4,"num_stages":3})", "quant_contract", 2},
        {64, 64, 128, R"({"num_warps":4,"num_stages":3})", "quant_contract", 2},
        {64, 128, 128, R"({"num_warps":8,"num_stages":3})", "quant_contract", 2},
        {128, 128, 64, R"({"num_warps":8,"num_stages":3})", "quant_contract", 2},
        {64, 64, 128, R"({"num_warps":4,"num_stages":3})", "scaled_contract"},
        {64, 128, 128, R"({"num_warps":8,"num_stages":3})", "scaled_contract"},
        {128, 128, 128, R"({"num_warps":8,"num_stages":3})", "scaled_contract"},
        {128, 128, 64, R"({"num_warps":8,"num_stages":3})", "contract"},
        {128, 256, 64, R"({"num_warps":8,"num_stages":4})", "contract", 2}};
  else
    matrix->tiles = {
        {2, 32, 256, R"({"num_warps":2,"num_stages":2})", "q4k_dot"},
        {4, 32, 256, R"({"num_warps":4,"num_stages":2})", "q4k_dot"},
        {8, 32, 256, R"({"num_warps":4,"num_stages":2})", "q4k_dot"},
        {16, 32, 32, R"({"num_warps":4,"num_stages":2})"},
        {32, 64, 32, R"({"num_warps":4,"num_stages":2})"},
        {32, 64, 64, R"({"num_warps":4,"num_stages":3})", "contract"},
        {64, 64, 64, R"({"num_warps":4,"num_stages":3})"},
        {64, 128, 64, R"({"num_warps":8,"num_stages":2})"},
        {32, 128, 128, R"({"num_warps":4,"num_stages":3})", "quant_contract", 2},
        {64, 64, 128, R"({"num_warps":4,"num_stages":3})", "quant_contract", 2},
        {64, 128, 128, R"({"num_warps":8,"num_stages":3})", "quant_contract", 2},
        {128, 128, 64, R"({"num_warps":8,"num_stages":3})", "quant_contract", 2},
        {128, 128, 64, R"({"num_warps":8,"num_stages":3})", "contract", 2},
        {64, 64, 128, R"({"num_warps":4,"num_stages":3})", "scaled_contract"},
        {64, 128, 128, R"({"num_warps":8,"num_stages":2})", "scaled_contract"},
        {128, 128, 128, R"({"num_warps":8,"num_stages":2})", "scaled_contract"}};
  if (auto err =
          profile.services.provide<intentdsl::block::MatrixTarget>(matrix))
    return std::move(err);
  caps->storeSectorBytes = 32;
  caps->nativePrivateHistogram = true;
  caps->privateBinCapacity = options.getInteger("shared_per_block");
  caps->maxExpandedReductionLanes = 65536;
  caps->independentWidths = {32, 64, 128};
  if(auto error=caps->bind(options))return error;
  struct NativeRow {
    unsigned id, rows, warps, stages;
  };
  static constexpr NativeRow hopper[] = {{0, 1, 4, 1}, {1, 2, 8, 1}, {2, 1, 8, 1}};
  static constexpr NativeRow blackwell[] = {{0, 1, 4, 1}, {1, 2, 4, 1}, {2, 1, 8, 1}};
  auto warp = options.getInteger("warp_size"),
       threads = options.getInteger("max_threads");
  if (!warp || !threads || *warp <= 0 || *threads <= 0)
    return llvm::createStringError(
        "expected local warp size and block thread limit");
  for (const auto &r : (*arch == 90 ? hopper : blackwell)) {
    if (r.warps > *threads / *warp)
      continue;
    std::string payload;
    llvm::raw_string_ostream os(payload);
    os << llvm::json::Value(llvm::json::Object{
        {"num_warps", int64_t(r.warps)}, {"num_stages", int64_t(r.stages)}});
    for(unsigned width=1; width<=8192; width*=2) {
      if (r.id == 2 && width < 2048)
        continue;
      unsigned rowId = r.id < 2 ? (width * 2 + r.id) : (262144u + width);
      caps->rows.push_back({rowId, r.rows, width, payload});
      if(r.id==0 && width<=128)
        for(unsigned group:{8u,16u})
          caps->rows.push_back({65536+32*width+group,group,width,payload});
    }
  }
  if(auto registers=options.getInteger("registers"))
    if(auto err=profile.services.facts.query("register_file",{*registers,"reg32","CUDA",llvm::json::Object{{"source","device properties"}}}))return err;
  if (auto err = profile.services.provide<intentdsl::block::OwnerMachine>(
          std::make_shared<intentdsl::block::OwnerMachine>(
              intentdsl::block::OwnerMachine{
                  128, 256,
                  options.getInteger("dynamic_shared_per_block").value_or(0),
                  32768, *arch == 90 ? 64 : 32,
                  *arch == 90
                      ? options.getInteger("registers").value_or(0) * 7 / 16
                      : options.getInteger("registers").value_or(0) * 3 / 8,
                  *arch == 90 ? 64 : 0, *arch == 90 ? 2 : 1})))
    return err;
  std::string cal = options.getString("calibration")
                        .value_or(*arch == 90 ? "sm90" : "sm120")
                        .str();
  if (auto err = intentdsl::bindPerformance(
          profile, options, "", cal, options.getInteger("sm_count").value_or(1)))
    return std::move(err);
  if (auto e = profile.services.provide<intentdsl::RowCapacity>(caps))
    return e;
  return profile;
}
} // namespace CI_UNIT_NAMESPACE
