#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#include "OwnerTiles.h"
namespace intentdsl::block {
void expandOwnerProgram(func::FuncOp fn, StringRef unit, OwnerAccess access) {
  fn.getContext()
      ->loadDialect<arith::ArithDialect, scf::SCFDialect, math::MathDialect>();
  auto &root = fn.front().front();
  if (intentdsl::kir::orderedOwners(fn) &&
      !root.hasAttr("block.column_prefixes")) {
    OwnerTiles tiles(fn, root, unit, access);
    (void)tiles.build();
    root.erase();
  }
}
} // namespace intentdsl::block
