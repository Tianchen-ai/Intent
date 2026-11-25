#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "ScaledPanels.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"

namespace CI_UNIT_NAMESPACE {
static bool supportsSharedDecode(mlir::Operation *domain, int64_t depth,
                                 const intentdsl::CallFacts &facts) {
  return domain->walk([&](mlir::Operation *op) {
    if (op->getName().stripDialect() != "quant_contract")
      return mlir::WalkResult::advance();
    auto *shape = facts.shapeOf(op->getOperand(0));
    if (shape && shape->size() == 3 &&
        sharedDecodeGroups(intentdsl::kir::quantFormatSpec(*op),
                           (*shape)[1].getAsInteger().value_or(0), depth))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  }).wasInterrupted();
}

static llvm::StringRef panelPlacement(intentdsl::block::MatrixAccess &a,
                                      int64_t participants,
                                      const intentdsl::dsa::MatrixStorage &ms) {
  bool union4 = ms.clusters == 4 && ms.coresPerCluster == 4 &&
                participants == ms.clusters * ms.coresPerCluster;
  if (a.tile[0] == 512 && a.tile[1] == 512)
    return union4 && ms.sramBytes >= 2097152 ? "shared-tile" : "";
  const auto *spec = intentdsl::kir::quantFormatSpec(a.dot);
  if (union4 && spec &&
      spec->roleLayout == intentdsl::kir::QuantRoleLayout::GroupedWeightRhs &&
      spec->packBits == 4 && spec->packFactor == 8 && a.tile[1] == 64 &&
      mlir::getElementTypeOrSelf(a.dot.getOperand(0)).isF16() &&
      mlir::getElementTypeOrSelf(a.dot.getOperand(2)).isF16() &&
      a.shape[0] >= a.tile[0] && a.shape[2] > 0 &&
      a.shape[2] % a.tile[2] == 0 && a.stride(a.dot.getOperand(0), 1) == 1 &&
      llvm::divideCeil(uint64_t(a.shape[1]), uint64_t(a.tile[1])) %
              ms.coresPerCluster == 0 &&
      ms.sramBytes >= 2 * a.tile[0] * a.tile[2] * int64_t(sizeof(uint16_t)))
    return "shared-lhs";
  return "private";
}
static int64_t matrixPrivateBytes(int64_t rows, int64_t columns, int64_t depth,
                                  llvm::StringRef computation) {
  if (rows == 512 && columns == 512) {
    // 2x2 cluster-cooperative tile: each of the 4 IPU cores owns a 256x256
    // quadrant (4 x [256, 64] accumulators + Split-M 2 x [128, depth] NRAM
    // staging + per-mode conversion/guard/LUT scratch).
    int64_t coreAcc = 4 * 256 * 64 * 4;
    int64_t coreLhs = 2 * 128 * depth * 2;
    int64_t extra = (computation == "quant_contract" ||
                     computation == "scaled_contract")
                        ? 131072
                        : 81920;
    return coreAcc + coreLhs + extra;
  }
  if (rows == 256 && columns == 64 && computation == "quant_contract") {
    // 256x64x128 grouped-RHS contraction: [256, 64] f32 acc + [256, depth] f16
    // lhs + [64, depth] decoded rhs + LUT/unpack work buffers.
    return 256 * 64 * 4 + 256 * depth * 2 + 64 * depth * 2 + 114688;
  }
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
  profile.reuse = compilerinfra::Reuse::Library;
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
                     {256, 64, 128, "{}", "quant_contract", 2},
                     {16, 64, 256, "{}", "contract"},
                     {32, 64, 256, "{}", "contract"},
                     {64, 64, 256, "{}", "contract"},
                     {128, 64, 256, "{}"},
                     {512, 512, 256, "{}", "contract", 2},
                     {512, 512, 128, "{}", "scaled_contract", 1, supportsSharedDecode},
                     {64, 64, 64, "{}", "window_contract"},
                     {32, 64, 64, "{}", "window_contract"}};
  auto wram = opts.getInteger("wram_bytes");
  auto sram = opts.getInteger("sram_bytes");
  // Only admit implementations that fit physical owners (and cluster SRAM for
  // 512x512 Union4 cooperative tiles). A row-only offline profile can still
  // compile its original row programs.
  llvm::erase_if(matrices->tiles, [&](auto &t) {
    bool isCluster = (t.m == 512 && t.n == 512);
    int64_t coreWramCols = isCluster ? (t.n / 2) : t.n;
    if (!wram || 2 * t.k * coreWramCols > *wram)
      return true;
    if (isCluster &&
        (!sram || *sram < 2097152 || *clusters != 4 || *cores != 4))
      return true;
    return matrixPrivateBytes(t.m, t.n, t.k, t.operation) >
           std::min<int64_t>(*nram / 3 * 2, 512 * 1024);
  });
  if (opts.getBoolean("cnnl_runtime").value_or(true))
    profile.features.insert("cnnl_runtime");
  matrices->estimate="exits/bangc::matrix-transfer";
  if(auto e=profile.services.provide<intentdsl::dsa::MatrixStorage>(
      std::make_shared<intentdsl::dsa::MatrixStorage>(intentdsl::dsa::MatrixStorage{
          matrixPrivateBytes, wram.value_or(0), sram.value_or(0), *clusters,
          *cores, "exits/bangc::cnnl-matmul-f16", "exits/bangc::cnnl-matmul-ftz",
          "exits/bangc::cnnl-matmul-cost", panelPlacement})))return e;
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
          (*clusters) * (*cores), sizeof(float), true, 16, 12, 8192,
          "exits/bangc::cnnl-transpose", "exits/bangc::cnnl-softmax",
          "exits/bangc::cnnl-layer-norm", "exits/bangc::cnnl-row-cost"});
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
