#include "Index.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
namespace CI_UNIT_NAMESPACE {
llvm::Expected<compilerinfra::TargetProfile>
capabilities(const llvm::json::Object &options) {
  compilerinfra::TargetProfile profile;
  if (options.empty())
    return profile;
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
  if(auto registers=options.getInteger("registers"))
    if(auto err=profile.services.facts.query("register_file",{*registers,"reg32","CUDA",llvm::json::Object{{"source","device properties"}}}))return err;
  if (auto err = profile.services.provide<intentdsl::block::OwnerMachine>(
          std::make_shared<intentdsl::block::OwnerMachine>(
              intentdsl::block::OwnerMachine{
                  32, 256,
                  options.getInteger("dynamic_shared_per_block").value_or(0),
                  *architecture == 90 ? 32768 : 24576,
                  *architecture == 90 ? 64 : 32,
                  *architecture == 90
                      ? options.getInteger("registers").value_or(0) / 2
                      : options.getInteger("registers").value_or(0) * 3 / 8,
                  *architecture == 90 ? 64 : 0})))
    return err;
  if(auto err=intentdsl::bindPerformance(profile,options,"",options.getString("calibration").value_or("").str(),options.getInteger("sm_count").value_or(1)))return std::move(err);
  if (auto e =
          profile.services.provide<intentdsl::cutile::ArrayIndices>(indices))
    return e;
  auto table = std::make_shared<intentdsl::RowCapacity>();
  table->storeSectorBytes = 32;
  table->globalAtomicBins=true;
  auto matrix=std::make_shared<intentdsl::block::MatrixTarget>();
  matrix->estimate="exits/cutile::matrix-issues";
  if(*architecture==90)matrix->tiles={{32,64,32,R"({"num_ctas":1,"occupancy":2,"opt_level":3})"},
                                      {64,128,64,R"({"num_ctas":1,"occupancy":1,"opt_level":3})"},
                                      {64,64,256,R"({"num_ctas":1,"occupancy":1,"opt_level":3})","scaled_contract"},
                                      {32,64,128,R"({"num_ctas":1,"occupancy":2,"opt_level":3})","scaled_contract"}};
  else matrix->tiles={{16,32,32,R"({"num_ctas":1,"occupancy":2,"opt_level":3})"},
                      {32,64,64,R"({"num_ctas":1,"occupancy":2,"opt_level":3})"}};
  if(auto err=profile.services.provide<intentdsl::block::MatrixTarget>(matrix))return std::move(err);
  table->privateBinCapacity=options.getInteger("shared_per_block");
  table->rectangularProjectedRows = true;
  // Bound tileiras expansion of repeated collectives. The limit admits a
  // single streamed 4096-wide row with 16 reductions; expanding that body
  // over two rows exceeded the native compilation budget on both targets.
  table->maxExpandedReductionLanes = 65536;
  table->independentWidths = {8, 16, 32, 64};
  if(auto error=table->bind(options))return error;
  // Wide rectangular transfers amortize a large resident tile. For generic
  // strided loads a smaller table also bounds native compile-time expansion.
  const unsigned maxWidth = indices->permitTma ? 4096 : 256;
  for (unsigned width = 1; width <= maxWidth; width *= 2) {
    for (unsigned rows : {1u, 2u, 8u, 16u}) {
      if(rows>2 && width>128) continue;
      llvm::SmallVector<unsigned, 2> occupancies;
      if (*architecture == 90) {
        occupancies.push_back(2u);
        if (rows == 1 && width >= 1024)
          occupancies.push_back(4u);
      } else {
        occupancies.push_back(std::min(rows, 2u));
        if (rows == 1 && width >= 1024)
          occupancies.push_back(2u);
      }
      for (auto [idx, occupancy] : llvm::enumerate(occupancies)) {
        std::string rowOptions;
        llvm::raw_string_ostream(rowOptions) << llvm::json::Value(
            llvm::json::Object{{"num_ctas", 1},
                               {"occupancy", occupancy},
                               {"opt_level", 3}});
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
