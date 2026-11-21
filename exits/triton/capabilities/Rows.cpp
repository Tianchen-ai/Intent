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
  auto upper = options.getInteger("width_bound");
  if (!upper || *upper < 0 || *upper > 8192 ||
      (*upper && !llvm::isPowerOf2_64(*upper)))
    return llvm::createStringError("invalid row width class");
  auto caps = std::make_shared<intentdsl::RowCapacity>();
  caps->upperBound = *upper;
  unsigned width = *upper ? *upper : 8192;
  // Opaque payloads contain only native options. The shared model sees no warp
  // count.
  caps->rows = {{0, 1, width, R"({"num_warps":4,"num_stages":1})"},
                {1, 2, width,
                 *arch == 90 ? R"({"num_warps":8,"num_stages":1})"
                             : R"({"num_warps":4,"num_stages":1})"}};
  auto warp = options.getInteger("warp_size"),
       threads = options.getInteger("max_threads");
  if (!warp || !threads || *warp <= 0 || *threads <= 0)
    return llvm::createStringError(
        "expected local warp size and block thread limit");
  llvm::erase_if(caps->rows, [&](const auto &r) {
    auto warps = (r.id == 1 && *arch == 90) ? 8 : 4;
    return warps > *threads / *warp;
  });
  if (auto e = profile.services.provide<intentdsl::RowCapacity>(caps))
    return e;
  return profile;
}
} // namespace CI_UNIT_NAMESPACE
