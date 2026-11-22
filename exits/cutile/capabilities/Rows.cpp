#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "Index.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
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
  indices->permitTma = options.getBoolean("allow_tma").value_or(false);
  if(auto registers=options.getInteger("registers"))
    if(auto err=profile.services.facts.query("register_file",{*registers,"reg32","CUDA",llvm::json::Object{{"source","device properties"}}}))return err;
  if(auto err=intentdsl::bindPerformance(profile,options,"",options.getString("calibration").value_or("").str(),options.getInteger("sm_count").value_or(1)))return std::move(err);
  if (auto e =
          profile.services.provide<intentdsl::cutile::ArrayIndices>(indices))
    return e;
  auto table = std::make_shared<intentdsl::RowCapacity>();
  table->globalAtomicBins=true;
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
      unsigned occupancy = *architecture == 90 ? (rows == 1 ? 2 : 1) : std::min(rows, 2u);
      std::string options;
      llvm::raw_string_ostream(options) << llvm::json::Value(llvm::json::Object{
          {"num_ctas", 1}, {"occupancy", occupancy}, {"opt_level", 3}});
      auto id = rows <= 2 ? 2 * width + rows : 65536u + 32 * width + rows;
      table->rows.push_back({id, rows, width, options});
    }
  }
  if (auto e = profile.services.provide<intentdsl::RowCapacity>(table))
    return e;
  return profile;
}
} // namespace CI_UNIT_NAMESPACE
