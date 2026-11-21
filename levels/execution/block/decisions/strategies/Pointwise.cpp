#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "mlir/IR/Builders.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void registerPointwise(Unit &u) {
  Strategy s;
  s.name = "pointwise";
  s.family = u.qualify("structure");
  s.propose = [](Operation *op, const TargetProfile &p,
                 AnalysisManager &) -> llvm::Expected<Attribute> {
    bool writes = false, collective = false;
    op->walk([&](Operation *o) {
      auto n = o->getName().stripDialect();
      writes |= n == "write";
      collective |= n == "sum" || n == "max_num";
    });
    if (!writes || collective)
      return Attribute{};
    auto *capacity = p.services.get<intentdsl::RowCapacity>();
    if (!capacity || capacity->rows.empty())
      return llvm::createStringError(
          "pointwise mapping needs registered tile rows");
    // Independent lanes require no cross-tile carry. Tile the inner extent,
    // including short tails, and use the same traversal for all widths.
    unsigned limit = 0, width = 0, previous = 0;
    for (auto &r : capacity->rows) {
      limit = std::max(limit, r.width);
      if (r.width >= capacity->observedWidth && (!width || r.width < width))
        width = r.width;
    }
    bool fits = bool(width);
    if (!fits)
      width = limit;
    for (auto &r : capacity->rows)
      if (r.width < width)
        previous = std::max(previous, r.width);
    Builder b(op->getContext());
    SmallVector<Attribute> rows;
    for (auto &r : capacity->rows)
      if ((r.width == width ||
           (!fits && llvm::is_contained(capacity->streamingWidths, r.width))) &&
          (cast<MemRefType>(op->getOperand(0).getType()).getRank() != 1 ||
           r.rows == 1))
        rows.push_back(b.getDictionaryAttr(
            {b.getNamedAttr("id", b.getI32IntegerAttr(r.id)),
             b.getNamedAttr("rows", b.getI32IntegerAttr(r.rows)),
             b.getNamedAttr("width", b.getI32IntegerAttr(r.width)),
             b.getNamedAttr("payload", b.getStringAttr(r.payload))}));
    return Attribute(b.getDictionaryAttr(
        {b.getNamedAttr("guard",
                        b.getDenseI64ArrayAttr(
                            {fits ? (previous ? int64_t(previous) + 1 : 0)
                                  : int64_t(limit) + 1,
                             fits ? int64_t(width) : -1})),
         b.getNamedAttr("topology",
                        b.getStringAttr(fits ? "resident" : "stream")),
         b.getNamedAttr("rows", b.getArrayAttr(rows))}));
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
