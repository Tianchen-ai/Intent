#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void registerPointwise(Unit &u,StringRef family) {
  Strategy strategy;
  bool indirect=family=="indirect";
  strategy.name = indirect?"indexed-rows":family=="permutation"?"rectangular-copy":"pointwise";
  strategy.family = u.qualify(indirect?"partition":family);
  auto propose = [indirect](Operation *op, const TargetProfile &p,
                        AnalysisManager &) -> llvm::Expected<Attribute> {
    bool hasWrite = false, hasSummary = false, hasGather=false;
    op->walk([&](Operation *o) {
      auto n = o->getName().stripDialect();
      hasWrite |= n == "write";
      hasGather |= n=="gather_row";
      hasSummary |= n == "sum" || n == "max_num" || n == "welford" || n=="scan_sum" || n=="ordered_sum" || n=="count_keys" || n=="column_sum";
    });
    if (!hasWrite || hasSummary || hasGather!=indirect)
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
  strategy.enumerate=[propose](Operation *op,const TargetProfile &p,AnalysisManager &a)->llvm::Expected<std::vector<StructuralCandidate>>{
    auto rows=propose(op,p,a);if(!rows)return rows.takeError();
    std::vector<StructuralCandidate> cs;if(!*rows)return cs;
    Builder b(op->getContext());
    for(auto walk:{"source","destination","tile"}){
      if(StringRef(walk)!="source" && !intentdsl::cpu::rectangularCopy(op))continue;
      cs.push_back({walk,walk,intentdsl::cpu::partition(b,walk,cast<ArrayAttr>(*rows)),{StringRef(walk)=="source"?"row-independent":"bijective-rectangular-copy"}});
    }
    return cs;
  };
  u.strategies.push_back({std::move(strategy), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
