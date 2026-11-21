#include "intentdsl/levels/execution/cpu/ir/Task.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void registerPointwise(Unit &u) {
  Strategy strategy;
  strategy.name = "pointwise";
  strategy.family = u.qualify("partition");
  strategy.propose = [](Operation *op, const TargetProfile &p,
                        AnalysisManager &) -> llvm::Expected<Attribute> {
    bool hasWrite = false, hasSummary = false;
    op->walk([&](Operation *o) {
      auto n = o->getName().stripDialect();
      hasWrite |= n == "write";
      hasSummary |= n == "sum" || n == "max_num";
    });
    if (!hasWrite || hasSummary)
      return Attribute{};
    auto *m = p.services.get<intentdsl::cpu::RowMachine>();
    if (!m)
      return llvm::createStringError("missing task partition capacity");
    bool widens = false, needsCoordinates = false;
    unsigned writes = 0;
    op->walk([&](Operation *o) {
      writes += o->getName().getStringRef() == "cpu.write";
      needsCoordinates |= o->getName().getStringRef() == "cpu.ordinal";
      if (o->getName().getStringRef() != "cpu.cast")
        return;
      auto from = dyn_cast<RankedTensorType>(o->getOperand(0).getType());
      auto to = dyn_cast<RankedTensorType>(o->getResult(0).getType());
      widens |= from && to &&
                from.getElementTypeBitWidth() < to.getElementTypeBitWidth();
    });
    if ((needsCoordinates || (m->boundWidening && (widens || writes > 1))) &&
        m->wideningPanel)
      return Attribute(m->wideningConfigurations(op->getContext()));
    return Attribute(m->configurations(op->getContext(), m->structuredRows));
  };
  u.strategies.push_back({std::move(strategy), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
