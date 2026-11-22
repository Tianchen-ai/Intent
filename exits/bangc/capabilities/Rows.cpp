#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"

namespace CI_UNIT_NAMESPACE {
llvm::Expected<compilerinfra::TargetProfile>
capabilities(const llvm::json::Object &opts) {
  compilerinfra::TargetProfile profile;
  if (opts.empty())
    return profile;
  auto isa = opts.getInteger("isa"),
       nram = opts.getInteger("nram_bytes"),
       clusters = opts.getInteger("clusters"), cores = opts.getInteger("cores");
  if (!isa || *isa != 372 || !nram || *nram <= 0 || !clusters ||
      !cores || *clusters <= 0 || *cores <= 0)
    return llvm::createStringError(
        "BANG C requires the MLU370 CNRT ISA, local capacity and core counts");
  auto caps = std::make_shared<intentdsl::RowCapacity>();
  if(auto error=caps->bind(opts))return error;
  caps->streamingWidths = {128, 512, 4096};
  for (unsigned w = 1; w <= 4096; w *= 2)
    for (unsigned r : {1u, 2u, 16u, 32u, 64u}) {
      if(r>16 && w>128)continue;
      auto tasks = (*clusters) * (*cores);
      auto id=r>16 ? 1048576u+128*w+r : w*32+r;
      caps->rows.push_back({id, r, w, std::to_string(tasks)});
    }
  auto resource =
      std::make_shared<intentdsl::dsa::LocalSupply>(intentdsl::dsa::LocalSupply{
          *nram, std::min<int64_t>(*nram/3*2,512*1024), 128, (*clusters) * (*cores), sizeof(float), true, 16, 12});
  caps->stagedInputBudget=std::min(resource->capacity,resource->budget);
  if(auto err=intentdsl::bindPerformance(profile,opts,"exits/bangc::dma",opts.getString("calibration").value_or("").str(),*clusters * *cores))return std::move(err);
  if (auto e = profile.services.provide<intentdsl::RowCapacity>(caps))
    return e;
  if (auto e = profile.services.provide<intentdsl::dsa::LocalSupply>(resource))
    return e;
  return profile;
}
} // namespace CI_UNIT_NAMESPACE
