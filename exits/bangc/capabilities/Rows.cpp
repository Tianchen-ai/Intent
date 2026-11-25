#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"

namespace CI_UNIT_NAMESPACE {
static int64_t matrixPrivateBytes(int64_t rows, int64_t columns, int64_t depth,
                                  llvm::StringRef computation) {
  // Upper bound shared by the supported convolution, record and scaled-dot
  // microkernels, including conversion and record scratch.
  return 256 + 4 * (rows * depth + 2 * depth * columns + 2 * rows * columns) +
         2 * 292 +
         (computation == "window_contract" ? 4 * depth * (columns + 3) : 0) +
         (computation == "quant_contract" ? 98304 : 0);
}
llvm::Expected<compilerinfra::TargetProfile>
capabilities(const llvm::json::Object &opts) {
  compilerinfra::TargetProfile profile;
  profile.services.calibrationProfile =
      opts.getString("calibration").value_or("mlu370").str();
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
  auto matrices=std::make_shared<intentdsl::block::MatrixTarget>();
  matrices->tiles = {{8, 64, 64, "{}"},
                     {16, 64, 128, "{}"},
                     {32, 64, 128, "{}"},
                     {64, 64, 128, "{}"},
                     {128, 64, 128, "{}", "quant_contract"},
                     {16, 64, 256, "{}", "contract"},
                     {32, 64, 256, "{}", "contract"},
                     {64, 64, 256, "{}", "contract"},
                     {128, 64, 256, "{}"},
                     {64, 64, 64, "{}", "window_contract"},
                     {32, 64, 64, "{}", "window_contract"}};
  auto wram=opts.getInteger("wram_bytes");
  // Only admit implementations that fit both physical owners. A row-only
  // offline profile can still compile its original row programs.
  llvm::erase_if(matrices->tiles, [&](auto &t) {
    return !wram || 2 * t.k * t.n > *wram ||
           matrixPrivateBytes(t.m, t.n, t.k, t.operation) >
               std::min<int64_t>(*nram / 3 * 2, 512 * 1024);
  });
  matrices->estimate="exits/bangc::matrix-transfer";
  if(auto e=profile.services.provide<intentdsl::dsa::MatrixStorage>(
      std::make_shared<intentdsl::dsa::MatrixStorage>(intentdsl::dsa::MatrixStorage{matrixPrivateBytes, wram.value_or(0)})))return e;
  if(auto e=profile.services.provide<intentdsl::block::MatrixTarget>(matrices))return e;
  if(auto error=caps->bind(opts))return error;
  caps->storeSectorBytes = 128;
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
          *nram, std::min<int64_t>(*nram / 3 * 2, 512 * 1024), 128,
          (*clusters) * (*cores), sizeof(float), true, 16, 12, 8192});
  caps->stagedInputBudget=std::min(resource->capacity,resource->budget);
  if (auto err = profile.services.provide<intentdsl::block::OwnerMachine>(
          std::make_shared<intentdsl::block::OwnerMachine>(
              intentdsl::block::OwnerMachine{1, 0})))
    return err;
  if(auto err=intentdsl::bindPerformance(profile,opts,"exits/bangc::dma",opts.getString("calibration").value_or("").str(),*clusters * *cores))return std::move(err);
  if (auto e = profile.services.provide<intentdsl::RowCapacity>(caps))
    return e;
  if (auto e = profile.services.provide<intentdsl::dsa::LocalSupply>(resource))
    return e;
  return profile;
}
} // namespace CI_UNIT_NAMESPACE
