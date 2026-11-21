#include "Index.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
namespace CI_UNIT_NAMESPACE {
llvm::Expected<compilerinfra::TargetProfile>
capabilities(const llvm::json::Object &options) {
  compilerinfra::TargetProfile profile;
  if (options.empty())
    return profile;
  auto architecture = options.getInteger("sm"),
       cols = options.getInteger("observed_width");
  if (!architecture || (*architecture != 90 && *architecture != 120) || !cols ||
      *cols < 0)
    return llvm::createStringError(
        "cuTile requires SM90 or SM120 and a nonnegative bound width");
  auto bits = options.getInteger("index_bits");
  if (bits != 32 && bits != 64)
    return llvm::createStringError(
        "cuTile requires a proven array index class");
  auto indices = std::make_shared<intentdsl::cutile::ArrayIndices>();
  indices->bits = *bits;
  indices->permitTma = options.getBoolean("allow_tma").value_or(false);
  if (auto e =
          profile.services.provide<intentdsl::cutile::ArrayIndices>(indices))
    return e;
  auto table = std::make_shared<intentdsl::RowCapacity>();
  table->observedWidth = *cols;
  for (unsigned width = 1; width <= 4096; width *= 2) {
    for (unsigned rows : {1u, 2u}) {
      unsigned occupancy = *architecture == 90 ? 3 - rows : rows;
      std::string options;
      llvm::raw_string_ostream(options) << llvm::json::Value(llvm::json::Object{
          {"num_ctas", 1}, {"occupancy", occupancy}, {"opt_level", 3}});
      table->rows.push_back({2 * width + rows, rows, width, options});
    }
  }
  if (auto e = profile.services.provide<intentdsl::RowCapacity>(table))
    return e;
  return profile;
}
} // namespace CI_UNIT_NAMESPACE
