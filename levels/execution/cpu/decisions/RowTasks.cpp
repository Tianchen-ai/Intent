#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/execution/cpu/ir/IndexedEffects.h"
#include "llvm/ADT/DenseSet.h"
#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::cpu {
ArrayAttr RowMachine::configurations(MLIRContext *ctx, bool whole) const {
  Builder b(ctx);
  SmallVector<Attribute> values;
  for (auto [r, c] : rows)
    values.push_back(b.getDictionaryAttr(
        {b.getNamedAttr("grain", b.getI64IntegerAttr(r)),
         b.getNamedAttr("columns", b.getI64IntegerAttr(whole ? 0 : c))}));
  return b.getArrayAttr(values);
}
ArrayAttr RowMachine::wideningConfigurations(MLIRContext *ctx) const {
  Builder b(ctx);
  SmallVector<Attribute> values;
  if (wideningPanel > 0)
    for (auto row : rows)
      values.push_back(b.getDictionaryAttr(
          {b.getNamedAttr("grain", b.getI64IntegerAttr(row.grain)),
           b.getNamedAttr("columns", b.getI64IntegerAttr(wideningPanel))}));
  return b.getArrayAttr(values);
}
ArrayAttr RowMachine::memberConfigurations(MLIRContext *ctx,
                                           bool widenOnly) const {
  Builder b(ctx);
  SmallVector<Attribute> values;
  llvm::SmallDenseSet<Attribute> seen;
  for (auto [r, c] : rows) {
    int64_t col = (widenOnly && wideningPanel > 0) ? wideningPanel
                                                   : std::max<int64_t>(1, c);
    int64_t raw = r * std::max<int64_t>(1, c);
    int64_t elemGrain = std::max<int64_t>(1, raw / col) * col;
    auto entry = b.getDictionaryAttr(
        {b.getNamedAttr("grain", b.getI64IntegerAttr(elemGrain)),
         b.getNamedAttr("columns", b.getI64IntegerAttr(col))});
    if (seen.insert(entry).second)
      values.push_back(entry);
  }
  return b.getArrayAttr(values);
}
bool RowMachine::accepts(ArrayAttr values) const {
  if (!values || values.empty())
    return false;
  auto first = dyn_cast<DictionaryAttr>(values[0]);
  auto column = first ? first.getAs<IntegerAttr>("columns") : IntegerAttr();
  if (!column)
    return false;
  bool whole = column.getInt() == 0;
  if (whole && !structuredRows)
    return false;
  auto *ctx = values.getContext();
  auto allowed = configurations(ctx, whole);
  auto fixed = wideningConfigurations(ctx);
  auto members = memberConfigurations(ctx, false);
  auto widenMembers = memberConfigurations(ctx, true);
  if (!fixed.empty() && llvm::all_of(values, [&](Attribute row) {
        return llvm::is_contained(fixed, row);
      }))
    allowed = fixed;
  else if (!whole && !members.empty() &&
           llvm::all_of(values, [&](Attribute row) {
             return llvm::is_contained(members, row);
           }))
    allowed = members;
  else if (!whole && !widenMembers.empty() &&
           llvm::all_of(values, [&](Attribute row) {
             return llvm::is_contained(widenMembers, row);
           }))
    allowed = widenMembers;
  llvm::SmallDenseSet<Attribute> seen;
  return llvm::all_of(values, [&](Attribute row) {
    return llvm::is_contained(allowed, row) && seen.insert(row).second;
  });
}
} // namespace intentdsl::cpu
namespace CI_UNIT_NAMESPACE {
void registerRowReduction(Unit &);
void registerPanelReduction(Unit &);
void registerTupleReduction(Unit &);
void registerScan(Unit &);
void registerIndirect(Unit &);
void registerSegment(Unit &);
void registerKeyReduction(Unit &);
void registerAtomics(Unit &);
void registerPrefixScatter(Unit &);
void registerPointwise(Unit &,StringRef);
bool supportsSweepUnroll(Operation *op) {
  auto fn = op->getParentOfType<func::FuncOp>();
  if (fn && fn->hasAttr("kir.owner_axis"))
    return false;
  bool blocked = false;
  op->walk([&](Operation *v) {
    auto n = v->getName().stripDialect();
    blocked |= n == "welford" || n == "scan_sum" || n == "uniform" ||
               n == "count_keys" || n == "column_sum" || n == "ordered_sum" ||
               n == "atomic_add" || n == "fill_ranges" || n == "unique_put" ||
               n == "write_offsets" || n == "prefix_scatter";
  });
  return !blocked;
}
llvm::Expected<std::vector<StructuralCandidate>>
enumerateReductionRows(Operation *op, const TargetProfile &profile,
                       bool structured, llvm::StringRef strategyId) {
  auto *machine = profile.services.get<intentdsl::cpu::RowMachine>();
  if (!machine)
    return llvm::createStringError("missing CPU capacity");
  bool reduces = false, ordinal = false, widens = false;
  op->walk([&](Operation *v) {
    auto n = v->getName().getStringRef();
    reduces |= n == "cpu.sum" || n == "cpu.max_num";
    ordinal |= n == "cpu.ordinal";
    if (n == "cpu.cast") {
      auto from = dyn_cast<RankedTensorType>(v->getOperand(0).getType());
      auto to = dyn_cast<RankedTensorType>(v->getResult(0).getType());
      widens |=
          from && to && from.getElementTypeBitWidth() < to.getElementTypeBitWidth();
    }
  });
  std::vector<StructuralCandidate> out;
  if (!reduces)
    return out;
  Builder b(op->getContext());
  ArrayAttr rows =
      ((ordinal || (widens && machine->boundWidening)) && machine->wideningPanel)
          ? machine->wideningConfigurations(op->getContext())
          : machine->configurations(
                op->getContext(),
                structured && machine->structuredRows &&
                    !op->getParentOfType<func::FuncOp>()->hasAttr(
                        "kir.owner_axis"));
  bool canUnroll = supportsSweepUnroll(op);
  for (int64_t u : machine->unrollFactors) {
    if (u > 1 && !canUnroll)
      continue;
    NamedAttrList layout(intentdsl::cpu::partition(b, "source", rows));
    if (machine->unrollFactors.size() > 1 || u > 1)
      layout.set("unroll", b.getI64IntegerAttr(u));
    std::string id =
        u > 1 ? (strategyId.str() + "-unroll" + std::to_string(u))
              : strategyId.str();
    out.push_back(
        {id, strategyId.str(), layout.getDictionary(op->getContext()), {}});
  }
  return out;
}
void registerPartition(Unit &u,bool permuted) {
  DecisionFamily partition;
  partition.name = permuted?"permutation":"partition";
  std::string carrier="cpu."+partition.name;
  partition.applies = [permuted](Operation *op) {
    return op->getName().getStringRef() == "cpu.parallel" && permuted==intentdsl::cpu::rectangularCopy(op);
  };
  partition.read = [carrier](Operation *op) { return op->getAttr(carrier); };
  partition.write = [carrier](Operation *op, Attribute a) {
    op->setAttr(carrier, a);
    return llvm::Error::success();
  };
  partition.verify = [](Operation *op, Attribute a,
                        const TargetProfile &p) -> llvm::Error {
    auto *machine = p.services.get<intentdsl::cpu::RowMachine>();
    auto d=dyn_cast<DictionaryAttr>(a);auto walk=d?d.getAs<StringAttr>("walk"):StringAttr();
    if(d && (d.get("key_extent") || d.get("batch_extent"))){
      auto &body=op->getRegion(0).front();auto plan=intentdsl::cpu::matchReservations(body);
      auto size=d.getAs<IntegerAttr>(d.get("key_extent")?"key_extent":"batch_extent");
      auto *facts=p.services.get<intentdsl::CallFacts>();
      if(!plan.update || !size || !facts)return llvm::createStringError("invalid keyed dispatch specialization");
      auto slot=cast<BlockArgument>(plan.update->getOperand(0)).getArgNumber();
      auto *shape=facts->arguments[slot].getAsObject()->getArray("shape");
      if(!shape || shape->size()!=1 || shape->front().getAsInteger()!=size.getInt() ||
         (d.contains("batch_extent") && size.getInt()>1024))return llvm::createStringError("invalid keyed dispatch extent");
      for(size_t i=0;i<facts->arguments.size();++i)for(size_t j=0;j<i;++j){
        auto *lhs=facts->arguments[i].getAsObject(),*rhs=facts->arguments[j].getAsObject();
        if((d.contains("batch_extent") || i==slot || j==slot) && lhs && rhs && *lhs->getArray("shape")==*rhs->getArray("shape"))
          return llvm::createStringError("keyed dispatch requires disjoint buffers");
      }
    }
    if (d && d.get("member_tasks")) {
      bool rank1 =
          cast<MemRefType>(op->getOperand(0).getType()).getRank() == 1 &&
          !op->getParentOfType<func::FuncOp>()->hasAttr("kir.owner_axis");
      bool hasOrderedSum = false, hasWrite = false, hasDisallowed = false;
      op->walk([&](Operation *v) {
        auto n = v->getName().stripDialect();
        hasOrderedSum |= n == "ordered_sum";
        hasWrite |= n == "write";
        hasDisallowed |= n == "sum" || n == "max_num" || n == "welford" ||
                         n == "scan_sum" || n == "count_keys" ||
                         n == "column_sum" || n == "gather_row" ||
                         n == "atomic_add" || n == "unique_put" ||
                         n == "pack_members" || n == "fill_ranges" ||
                         n == "reserve_tickets" || n == "distribute_tickets" ||
                         n == "write_offsets";
      });
      if (!rank1 || walk != "source" ||
          (!hasOrderedSum && (!hasWrite || hasDisallowed)))
        return llvm::createStringError(
            "flat member partition needs independent rank-1 member outputs");
    }
    int64_t unroll = 1;
    if (d && d.get("unroll")) {
      auto uAttr = d.getAs<IntegerAttr>("unroll");
      if (!uAttr || !machine || !machine->acceptsUnroll(uAttr.getInt()) ||
          (uAttr.getInt() > 1 &&
           (walk != "source" || !supportsSweepUnroll(op))))
        return llvm::createStringError("illegal CPU sweep unroll factor");
      unroll = uAttr.getInt();
    }
    if (!walk || (walk!="source" && ( (walk!="destination" && walk!="tile") || !intentdsl::cpu::rectangularCopy(op))) || !d.getAs<IntegerAttr>("tile") || d.getAs<IntegerAttr>("tile").getInt()!=(walk=="tile"?32:1) || !machine || !machine->accepts(d.getAs<ArrayAttr>("rows")))
      return llvm::createStringError("illegal CPU work partition");
    return llvm::Error::success();
  };
  partition.verifyCandidate=[](Operation *op,const StructuralCandidate &c,const TargetProfile &)->llvm::Error{
    auto d=cast<DictionaryAttr>(c.value);
    auto walk=d.getAs<StringAttr>("walk");
    int64_t unroll = d.getAs<IntegerAttr>("unroll")
                         ? d.getAs<IntegerAttr>("unroll").getInt()
                         : 1;
    std::string suffix =
        unroll > 1 ? ("-unroll" + std::to_string(unroll)) : "";
    if(c.id=="range-reservations" && c.topology==c.id && c.preconditions==std::vector<std::string>{"independent-relaxed-unit-tickets"} &&
       d.getAs<IntegerAttr>("batch_extent"))return llvm::Error::success();
    if(c.id=="key-owners" && c.topology==c.id && walk=="source" && c.preconditions==std::vector<std::string>{"disjoint-key-ranges"} &&
       d.getAs<IntegerAttr>("key_extent") && d.getAs<IntegerAttr>("key_extent").getInt()>0)
      return llvm::Error::success();
    if(c.id=="serial-tickets" && c.topology==c.id && walk=="source" && d.contains("single_dispatcher") &&
       c.preconditions==std::vector<std::string>{"relaxed-keyed-updates"} &&
       op->walk([](Operation *v){return v->getName().stripDialect()=="atomic_add"?WalkResult::interrupt():WalkResult::advance();}).wasInterrupted())return llvm::Error::success();
    if(c.id=="levels/execution/cpu::row-scan" && c.topology==c.id && walk=="source" && c.preconditions.empty() &&
       op->walk([](Operation *v){return v->getName().getStringRef()=="cpu.scan_sum"?WalkResult::interrupt():WalkResult::advance();}).wasInterrupted())return llvm::Error::success();
    if(c.topology==walk.getValue() && c.id==(walk.str()+suffix) && c.preconditions==std::vector<std::string>{walk=="source"?"row-independent":"bijective-rectangular-copy"})return llvm::Error::success();
    if(c.topology.find("reduction")!=std::string::npos && c.id==(c.topology+suffix) && walk=="source" && c.preconditions.empty())return llvm::Error::success();
    return llvm::createStringError("CPU topology disagrees with its parameter domain");};
  u.families.push_back(
      {{partition, "cpu@formed", "cpu@formed"}, COMPILERINFRA_SOURCE});
  if(!permuted){registerIndirect(u);registerAtomics(u);registerScan(u);registerPrefixScatter(u);registerSegment(u);registerKeyReduction(u);registerTupleReduction(u);registerRowReduction(u);registerPanelReduction(u);}
  auto family=permuted?"permutation":"partition";registerPointwise(u,family);
  u.realizers.push_back(
      {{permuted?"permute-rows":"group-rows", u.qualify(family),
        [carrier](Operation *op, Attribute a, const TargetProfile &, RewriterBase &) {
          auto d=cast<DictionaryAttr>(a);
          if(d.get("member_tasks"))op->setAttr("cpu.member_tasks",d.get("member_tasks"));
          if(d.get("single_dispatcher"))op->setAttr("cpu.single_dispatcher",d.get("single_dispatcher"));
          if(auto keys=d.get("key_extent"))op->setAttr("cpu.key_extent",keys);
          if(auto bins=d.get("batch_extent"))op->setAttr("cpu.batch_extent",bins);
          if(auto unroll=d.get("unroll"))op->setAttr("cpu.unroll",unroll);
          op->setAttr("cpu.row_groups",d.get("rows"));op->setAttr("cpu.walk",d.get("walk"));op->setAttr("cpu.tile",d.get("tile"));
          op->removeAttr(carrier);
          return llvm::Error::success();
        }},
       COMPILERINFRA_SOURCE});
}
void rowTasks(Unit &u){registerPartition(u,false);registerPartition(u,true);}
} // namespace CI_UNIT_NAMESPACE
