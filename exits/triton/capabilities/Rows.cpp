#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
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
    for(unsigned width=1; width<=8192; width*=2)
      caps->rows.push_back({width*2+r.id, r.rows, width, payload});
  }
  if (auto e = profile.services.provide<intentdsl::RowCapacity>(caps))
    return e;
  return profile;
}
} // namespace CI_UNIT_NAMESPACE
