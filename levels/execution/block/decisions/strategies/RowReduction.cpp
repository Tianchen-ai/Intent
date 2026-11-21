#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "mlir/IR/Builders.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerRowReduction(Unit &unit) {
  Strategy strategy;
  strategy.name = "row-reduction";
  strategy.family = unit.qualify("structure");
  strategy.propose = [](Operation *op, const TargetProfile &target,
                        AnalysisManager &) -> llvm::Expected<Attribute> {
    bool reduces = false;
    op->walk([&](Operation *v) {
      auto n = v->getName().getStringRef();
      reduces |= n == "block.sum" || n == "block.max_num";
    });
    if (!reduces)
      return Attribute{};
    auto *caps = target.services.get<intentdsl::RowCapacity>();
    if (!caps || caps->rows.empty())
      return llvm::createStringError(
          "row reduction requires a bound GPU capability table");
    Builder b(op->getContext());
    bool resident = caps->upperBound != 0;
    SmallVector<Attribute> rows;
    for (const auto &r : caps->rows) {
      if (resident && r.width < caps->upperBound)
        continue;
      rows.push_back(b.getDictionaryAttr(
          {b.getNamedAttr("id", b.getI32IntegerAttr(r.id)),
           b.getNamedAttr("rows", b.getI32IntegerAttr(r.rows)),
           b.getNamedAttr("width", b.getI32IntegerAttr(r.width)),
           b.getNamedAttr("payload", b.getStringAttr(r.payload))}));
    }
    if (rows.empty())
      return llvm::createStringError("no legal row configurations");
    return Attribute(b.getDictionaryAttr(
        {b.getNamedAttr("topology",
                        b.getStringAttr(resident ? "resident" : "stream")),
         b.getNamedAttr("rows", b.getArrayAttr(rows))}));
  };
  unit.strategies.push_back({std::move(strategy), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
