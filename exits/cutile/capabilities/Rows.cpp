#include "Index.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include <dlfcn.h>
namespace CI_UNIT_NAMESPACE {
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
llvm::Expected<compilerinfra::TargetProfile>
capabilities(const llvm::json::Object &options) {
  compilerinfra::TargetProfile profile;
  if (options.empty() || !options.getInteger("sm")) {
    profile.services.calibrationProfile =
        options.getString("calibration").value_or(probeCudaCalibration()).str();
    return profile;
  }
  auto architecture = options.getInteger("sm");
  if (!architecture || (*architecture != 90 && *architecture != 120))
    return llvm::createStringError(
        "cuTile requires SM90 or SM120 and a nonnegative bound width");
  auto bits = options.getInteger("index_bits");
  if (bits != 32 && bits != 64)
    return llvm::createStringError(
        "cuTile requires a proven array index class");
  auto indices = std::make_shared<intentdsl::cutile::ArrayIndices>();
  indices->bits = *bits;
  indices->scaledMma=*architecture>=100;
  indices->permitTma = options.getBoolean("allow_tma").value_or(false);
  indices->preferRectRowReads = (*architecture == 90);
  int64_t warpSize = options.getInteger("warp_size").value_or(32);
  int64_t exchangeW = *architecture == 90 ? 64 : 32;
  int64_t residentEl = 256;
  if(auto registers=options.getInteger("registers"))
    if(auto err=profile.services.facts.query("register_file",{*registers,"reg32","CUDA",llvm::json::Object{{"source","device properties"}}}))return err;
  if (auto err = profile.services.provide<intentdsl::block::OwnerMachine>(
          std::make_shared<intentdsl::block::OwnerMachine>(
              intentdsl::block::OwnerMachine{
                  128, int(residentEl),
                  options.getInteger("dynamic_shared_per_block").value_or(0),
                  *architecture == 90 ? 32768 : 24576, exchangeW,
                  *architecture == 90
                      ? options.getInteger("registers").value_or(0) / 2
                      : options.getInteger("registers").value_or(0) * 3 / 8,
                  *architecture == 90 ? 64 : 0,
                  /*maxWarpgroups=*/1,
                  /*maxGroupReads=*/16,
                  /*nativeTileGather=*/false,
                  warpSize,
                  /*scatterPanelWidth=*/64,
                  /*histogramPanelWidth=*/residentEl * 2,
                  /*minHistogramPanelWidth=*/64})))
    return err;
  std::string cal = options.getString("calibration")
                        .value_or(*architecture == 90 ? "sm90" : "sm120")
                        .str();
  if (auto err = intentdsl::bindPerformance(
          profile, options, "", cal, options.getInteger("sm_count").value_or(1)))
    return std::move(err);
  if (auto e =
          profile.services.provide<intentdsl::cutile::ArrayIndices>(indices))
    return e;
  auto table = std::make_shared<intentdsl::RowCapacity>();
  table->storeSectorBytes = 32;
  table->globalAtomicBins=true;
  auto matrix=std::make_shared<intentdsl::block::MatrixTarget>();
  matrix->estimate="exits/cutile::matrix-issues";
  matrix->supportsSplitGroupedRhs = true;
  matrix->flatSplitDequant = false;
  matrix->splitDequantWriteScale = (*architecture == 90 ? 1.5 : 3.5);
  matrix->l2CacheBytes = options.getInteger("l2_cache_bytes").value_or(0);
  matrix->smMmaOpsPerNs = (*architecture == 90 ? 5200.0 : 1800.0);
  matrix->libraryGemmImpl = "exits/cutile::cublas-gemm";
  matrix->librarySmMmaOpsPerNs = (*architecture == 90 ? 5500.0 : 1700.0);
  matrix->implementations = {
      {"contract", {"exits/cutile::native-dot"}},
      {"window_contract", {"exits/cutile::native-dot"}},
      {"scaled_contract",
       indices->scaledMma
           ? std::vector<std::string>{"exits/cutile::scaled-mma",
                                      "exits/cutile::native-dot",
                                      "exits/cutile::row-collective"}
           : std::vector<std::string>{"exits/cutile::native-dot",
                                      "exits/cutile::row-collective"}},
      {"q4k_dot", {"exits/cutile::row-collective"}}};
  if (*architecture == 90)
    matrix->tiles = {
        {2, 32, 256, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "q4k_dot"},
        {4, 32, 256, R"({"num_ctas":1,"occupancy":4,"opt_level":3})",
         "q4k_dot"},
        {8, 32, 256, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "q4k_dot"},
        {16, 32, 32, R"({"num_ctas":1,"occupancy":2,"opt_level":3})"},
        {32, 64, 32, R"({"num_ctas":1,"occupancy":2,"opt_level":3})"},
        {32, 64, 64, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "contract"},
        {64, 32, 64, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "window_contract"},
        {64, 128, 64, R"({"num_ctas":1,"occupancy":1,"opt_level":3})"},
        {32, 128, 128, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "quant_contract", 2},
        {64, 64, 64, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "quant_contract", 2},
        {64, 64, 128, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "quant_contract", 2},
        {64, 128, 128, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "quant_contract", 2},
        {128, 64, 128, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "quant_contract", 2},
        {128, 128, 64, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "quant_contract", 2},
        {128, 128, 64, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "contract"},
        {64, 64, 256, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "scaled_contract"},
        {64, 128, 128, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "scaled_contract"},
        {128, 128, 128, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "scaled_contract"},
        {32, 64, 128, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "scaled_contract"}};
  else
    matrix->tiles = {
        {2, 32, 256, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "q4k_dot"},
        {4, 32, 256, R"({"num_ctas":1,"occupancy":4,"opt_level":3})",
         "q4k_dot"},
        {8, 32, 256, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "q4k_dot"},
        {16, 32, 32, R"({"num_ctas":1,"occupancy":2,"opt_level":3})"},
        {32, 64, 64, R"({"num_ctas":1,"occupancy":2,"opt_level":3})"},
        {64, 32, 64, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "window_contract"},
        {64, 64, 64, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "contract"},
        {64, 128, 64, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "contract"},
        {32, 128, 128, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "quant_contract", 2},
        {64, 64, 64, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "quant_contract", 2},
        {64, 64, 128, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "quant_contract", 2},
        {64, 128, 64, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "quant_contract", 2},
        {64, 128, 128, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "quant_contract", 2},
        {128, 64, 128, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "quant_contract", 2},
        {128, 128, 64, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "quant_contract", 2},
        {128, 128, 64, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "contract", 2},
        {64, 64, 256, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "scaled_contract"},
        {64, 128, 128, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "scaled_contract"},
        {128, 128, 128, R"({"num_ctas":1,"occupancy":1,"opt_level":3})",
         "scaled_contract"},
        {32, 64, 128, R"({"num_ctas":1,"occupancy":2,"opt_level":3})",
         "scaled_contract"}};
  if (auto err =
          profile.services.provide<intentdsl::block::MatrixTarget>(matrix))
    return std::move(err);
  table->privateBinCapacity=options.getInteger("shared_per_block");
  table->rectangularProjectedRows = true;
  // Bound tileiras expansion of repeated collectives. The limit admits a
  // single streamed 4096-wide row with 16 reductions; expanding that body
  // over two rows exceeded the native compilation budget on both targets.
  table->maxExpandedReductionLanes = 65536;
  table->independentWidths = {8, 16, 32, 64, 128};
  if (auto error = table->bind(options))
    return error;
  // Wide rectangular transfers amortize a large resident tile. For generic
  // strided loads a smaller table also bounds native compile-time expansion.
  const unsigned maxWidth = indices->permitTma ? 4096 : 256;
  const unsigned warpU = unsigned(std::max<int64_t>(1, warpSize));
  const unsigned maxMultiRowElems = warpU * 32u;
  for (unsigned width = 1; width <= maxWidth; width *= 2) {
    for (unsigned rows : {1u, 2u, 8u, 16u}) {
      if (rows > 2 && width > 128)
        continue;
      llvm::SmallVector<unsigned, 2> occupancies;
      bool narrowMultiRow =
          rows == 8 && width >= warpU && rows * width <= maxMultiRowElems;
      if (*architecture == 90) {
        occupancies.push_back(2u);
        if ((rows == 1 && ((width >= 32 && width <= 256) || width >= 1024)) ||
            narrowMultiRow)
          occupancies.push_back(4u);
      } else {
        occupancies.push_back(
            rows == 1 && ((width >= 32 && width <= 256) || width >= 1024)
                ? 2u
                : std::min(rows, 2u));
        if ((rows == 1 && ((width >= 32 && width <= 256) || width >= 512)) ||
            narrowMultiRow)
          occupancies.push_back(
              (rows == 1 &&
               ((width >= 32 && width <= 256) || width >= 1024)) ||
                      narrowMultiRow
                  ? 4u
                  : 2u);
      }
      for (auto [idx, occupancy] : llvm::enumerate(occupancies)) {
        std::string rowOptions;
        llvm::raw_string_ostream(rowOptions) << llvm::json::Value(
            llvm::json::Object{
                {"num_ctas", 1}, {"occupancy", occupancy}, {"opt_level", 3}});
        auto baseId = rows <= 2 ? 2 * width + rows : 65536u + 32 * width + rows;
        auto id = idx == 0 ? baseId : 131072u + baseId;
        table->rows.push_back({id, rows, width, rowOptions});
      }
    }
  }
  if (auto e = profile.services.provide<intentdsl::RowCapacity>(table))
    return e;
  return profile;
}
} // namespace CI_UNIT_NAMESPACE
