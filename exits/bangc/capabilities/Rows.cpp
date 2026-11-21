#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"

namespace CI_UNIT_NAMESPACE {
llvm::Expected<compilerinfra::TargetProfile>
capabilities(const llvm::json::Object &opts) {
  compilerinfra::TargetProfile profile;
  if (opts.empty())
    return profile;
  auto isa = opts.getInteger("isa"), n = opts.getInteger("observed_width"),
       nram = opts.getInteger("nram_bytes"),
       clusters = opts.getInteger("clusters"), cores = opts.getInteger("cores");
  if (!isa || *isa != 372 || !n || *n < 0 || !nram || *nram <= 0 || !clusters ||
      !cores || *clusters <= 0 || *cores <= 0)
    return llvm::createStringError(
        "BANG C requires the MLU370 CNRT ISA, local capacity and core counts");
  auto caps = std::make_shared<intentdsl::RowCapacity>();
  caps->observedWidth = *n;
  for (unsigned w = 1; w <= 4096; w *= 2)
    for (unsigned r : {1u, 2u, 16u}) {
      auto tasks = (*clusters) * (*cores);
      caps->rows.push_back({w * 32 + r, r, w, std::to_string(tasks)});
    }
  auto resource =
      std::make_shared<intentdsl::dsa::LocalSupply>(intentdsl::dsa::LocalSupply{
          *nram, 128 * 1024, 128, (*clusters) * (*cores), sizeof(float), true});
  if (auto e = profile.services.provide<intentdsl::RowCapacity>(caps))
    return e;
  if (auto e = profile.services.provide<intentdsl::dsa::LocalSupply>(resource))
    return e;
  return profile;
}
} // namespace CI_UNIT_NAMESPACE
