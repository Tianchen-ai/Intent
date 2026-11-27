#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
bool supportsSweepUnroll(Operation *op);
void registerPointwise(Unit &u,StringRef family) {
  Strategy strategy;
  bool indirect=family=="indirect";
  strategy.name = indirect?"indexed-rows":family=="permutation"?"rectangular-copy":"pointwise";
  strategy.family = u.qualify(indirect?"partition":family);
  auto isRank1Independent = [indirect](Operation *op) {
    if (indirect ||
        cast<MemRefType>(op->getOperand(0).getType()).getRank() != 1 ||
        op->getParentOfType<func::FuncOp>()->hasAttr("kir.owner_axis"))
      return false;
    bool hasCrossMember = false;
    op->walk([&](Operation *o) {
      auto n = o->getName().stripDialect();
      hasCrossMember |= n == "atomic_add" || n == "unique_put" ||
                        n == "pack_members" || n == "fill_ranges" ||
                        n == "reserve_tickets" || n == "distribute_tickets" ||
                        n == "write_offsets" || n == "encode_q8k";
    });
    return !hasCrossMember;
  };
  auto propose = [indirect, isRank1Independent](
                     Operation *op, const TargetProfile &p,
                     AnalysisManager &) -> llvm::Expected<Attribute> {
    bool hasWrite = false, hasSummary = false, hasGather=false;
    op->walk([&](Operation *o) {
      auto n = o->getName().stripDialect();
      hasWrite |= n == "write" || n == "encode_q8k";
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
      writes += (o->getName().getStringRef() == "cpu.write" ||
                 o->getName().getStringRef() == "cpu.encode_q8k");
      needsCoordinates |= o->getName().getStringRef() == "cpu.ordinal";
      if (o->getName().getStringRef() != "cpu.cast")
        return;
      auto from = dyn_cast<RankedTensorType>(o->getOperand(0).getType());
      auto to = dyn_cast<RankedTensorType>(o->getResult(0).getType());
      widens |= from && to &&
                from.getElementTypeBitWidth() < to.getElementTypeBitWidth();
    });
    bool widen = (needsCoordinates || (m->boundWidening && (widens || writes > 1))) &&
                 m->wideningPanel;
    if (isRank1Independent(op))
      return Attribute(m->memberConfigurations(op->getContext(), widen));
    if (widen)
      return Attribute(m->wideningConfigurations(op->getContext()));
    return Attribute(m->configurations(op->getContext(), m->structuredRows));
  };
  strategy.enumerate=[propose, isRank1Independent, indirect, permuted = (family == "permutation")](Operation *op,const TargetProfile &p,AnalysisManager &a)->llvm::Expected<std::vector<StructuralCandidate>>{
    auto rows=propose(op,p,a);if(!rows)return rows.takeError();
    std::vector<StructuralCandidate> cs;if(!*rows)return cs;
    Builder b(op->getContext());
    bool memberTasks = isRank1Independent(op);
    auto *m = p.services.get<intentdsl::cpu::RowMachine>();
    bool canUnroll = !indirect && !permuted && supportsSweepUnroll(op);
    for(auto walk:{"source","destination","tile"}){
      if(StringRef(walk)!="source" && !intentdsl::cpu::rectangularCopy(op))continue;
      if (StringRef(walk) == "source" && m && !permuted && !indirect) {
        for (int64_t uf : m->unrollFactors) {
          if (uf > 1 && !canUnroll)
            continue;
          NamedAttrList layout(
              intentdsl::cpu::partition(b, walk, cast<ArrayAttr>(*rows)));
          if (memberTasks)
            layout.set("member_tasks", b.getUnitAttr());
          if (m->unrollFactors.size() > 1 || uf > 1)
            layout.set("unroll", b.getI64IntegerAttr(uf));
          std::string id =
              uf > 1 ? (std::string(walk) + "-unroll" + std::to_string(uf))
                     : std::string(walk);
          cs.push_back({id,
                        walk,
                        layout.getDictionary(op->getContext()),
                        {"row-independent"}});
        }
        continue;
      }
      NamedAttrList layout(intentdsl::cpu::partition(b,walk,cast<ArrayAttr>(*rows)));
      if(memberTasks && StringRef(walk)=="source")
        layout.set("member_tasks",b.getUnitAttr());
      cs.push_back({walk,walk,layout.getDictionary(op->getContext()),{StringRef(walk)=="source"?"row-independent":"bijective-rectangular-copy"}});
    }
    return cs;
  };
  u.strategies.push_back({std::move(strategy), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
