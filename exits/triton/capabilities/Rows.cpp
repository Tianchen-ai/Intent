#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "llvm/Support/MathExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &options) {
  TargetProfile profile;
  auto arch = options.getInteger("sm");
  if (!arch)
    return profile; // Static layout has no device binding.
  if (*arch != 90 && *arch != 120)
    return llvm::createStringError("unsupported Triton GPU architecture");
  auto caps = std::make_shared<intentdsl::RowCapacity>();
  caps->globalAtomicBins=true;
  auto matrix=std::make_shared<intentdsl::block::MatrixTarget>();
  matrix->estimate="exits/triton::matrix-issues";
  if(*arch==90)matrix->tiles={{32,64,32,R"({"num_warps":4,"num_stages":3})"},
                              {64,64,32,R"({"num_warps":4,"num_stages":3})"},
                              {64,128,64,R"({"num_warps":8,"num_stages":3})"}};
  else matrix->tiles={{16,32,32,R"({"num_warps":4,"num_stages":2})"},
                      {32,64,32,R"({"num_warps":4,"num_stages":2})"},
                      {64,64,64,R"({"num_warps":4,"num_stages":2})"}};
  if(auto err=profile.services.provide<intentdsl::block::MatrixTarget>(matrix))return std::move(err);
  caps->nativePrivateHistogram=true;
  caps->privateBinCapacity=options.getInteger("shared_per_block");
  caps->independentWidths = {32};
  if(auto error=caps->bind(options))return error;
  struct NativeRow {
    unsigned id, rows, warps, stages;
  };
  static constexpr NativeRow hopper[] = {{0, 1, 4, 1}, {1, 2, 8, 1}};
  static constexpr NativeRow blackwell[] = {{0, 1, 4, 1}, {1, 2, 4, 1}};
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
      caps->rows.push_back({width*2+r.id, r.rows, width, payload});
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
                  32, 256,
                  options.getInteger("dynamic_shared_per_block").value_or(0), 0,
                  *arch == 90 ? 32 : 0,
                  *arch == 90 ? options.getInteger("registers").value_or(0) / 2
                              : 0,
                  *arch == 90 ? 64 : 0})))
    return err;
  if(auto err=intentdsl::bindPerformance(profile,options,"",options.getString("calibration").value_or("").str(),options.getInteger("sm_count").value_or(1)))return std::move(err);
  if (auto e = profile.services.provide<intentdsl::RowCapacity>(caps))
    return e;
  return profile;
}
} // namespace CI_UNIT_NAMESPACE
