#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "mlir/IR/Builders.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerRowReduction(Unit &unit) {
  for (StringRef mode : {"row-reduction", "resident-row", "stream-row"}) {
  Strategy strategy;
  strategy.name = mode.str();
  strategy.family = unit.qualify("structure");
  strategy.propose = [mode](Operation *op, const TargetProfile &target,
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
    unsigned width=0, limit=0, previous=0;
    for(auto &r:caps->rows) {
      limit=std::max(limit,r.width);
      if(r.width>=caps->observedWidth && (!width || r.width<width)) width=r.width;
    }
    bool resident = width && mode!="stream-row";
    if(!resident && mode=="resident-row") return Attribute{};
    if(!resident) width=limit;
    for(auto &r:caps->rows) if(r.width<width)previous=std::max(previous,r.width);
    int64_t lo=resident?(previous?previous+1:0):(mode=="stream-row"?0:limit+1);
    auto guard=b.getDenseI64ArrayAttr({lo,resident?int64_t(width):-1});
    SmallVector<Attribute> rows;
    for (const auto &r : caps->rows) {
      if (r.width != width)
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
        {b.getNamedAttr("guard",guard),b.getNamedAttr("topology",
                        b.getStringAttr(resident ? "resident" : "stream")),
         b.getNamedAttr("rows", b.getArrayAttr(rows))}));
  };
  unit.strategies.push_back({std::move(strategy), COMPILERINFRA_SOURCE});
  }
}
} // namespace CI_UNIT_NAMESPACE
