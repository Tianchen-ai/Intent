#include "intentdsl/levels/execution/cpu/ir/Task.h"
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
  auto allowed = configurations(values.getContext(), whole);
  auto fixed = wideningConfigurations(values.getContext());
  if (!fixed.empty() && llvm::all_of(values, [&](Attribute row) {
        return llvm::is_contained(fixed, row);
      }))
    allowed = fixed;
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
void registerPointwise(Unit &,StringRef);
llvm::Expected<Attribute> proposeRows(Operation *op,const TargetProfile &profile,bool structured){
  auto *machine=profile.services.get<intentdsl::cpu::RowMachine>();
  if(!machine)return llvm::createStringError("missing CPU capacity");
  bool reduces=false, ordinal=false, widens=false;
  op->walk([&](Operation *v){auto n=v->getName().getStringRef();reduces |= n=="cpu.sum" || n=="cpu.max_num";
    ordinal |= n=="cpu.ordinal";
    if(n=="cpu.cast"){
      auto from=dyn_cast<RankedTensorType>(v->getOperand(0).getType());
      auto to=dyn_cast<RankedTensorType>(v->getResult(0).getType());
      widens |= from && to && from.getElementTypeBitWidth()<to.getElementTypeBitWidth();
    }});
  if(!reduces)return Attribute{};
  Builder b(op->getContext());
  if ((ordinal || (widens && machine->boundWidening)) && machine->wideningPanel)
    return Attribute(intentdsl::cpu::partition(b,"source",machine->wideningConfigurations(op->getContext())));
  bool projectedOwner = op->getParentOfType<func::FuncOp>()->hasAttr("kir.owner_axis");
  return Attribute(intentdsl::cpu::partition(b,"source",machine->configurations(op->getContext(),structured && machine->structuredRows && !projectedOwner)));
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
    if (!walk || (walk!="source" && ( (walk!="destination" && walk!="tile") || !intentdsl::cpu::rectangularCopy(op))) || !d.getAs<IntegerAttr>("tile") || d.getAs<IntegerAttr>("tile").getInt()!=(walk=="tile"?32:1) || !machine || !machine->accepts(d.getAs<ArrayAttr>("rows")))
      return llvm::createStringError("illegal CPU work partition");
    return llvm::Error::success();
  };
  partition.verifyCandidate=[](Operation*,const StructuralCandidate &c,const TargetProfile &)->llvm::Error{
    auto walk=cast<DictionaryAttr>(c.value).getAs<StringAttr>("walk");
    if(c.id==c.topology && c.topology==walk.getValue() && c.preconditions==std::vector<std::string>{walk=="source"?"row-independent":"bijective-rectangular-copy"})return llvm::Error::success();
    if(c.id==c.topology && c.id.find("reduction")!=std::string::npos && walk=="source" && c.preconditions.empty())return llvm::Error::success();
    return llvm::createStringError("CPU topology disagrees with its parameter domain");};
  u.families.push_back(
      {{partition, "cpu@formed", "cpu@formed"}, COMPILERINFRA_SOURCE});
  if(!permuted){registerTupleReduction(u);registerRowReduction(u);registerPanelReduction(u);}
  auto family=permuted?"permutation":"partition";registerPointwise(u,family);
  u.realizers.push_back(
      {{permuted?"permute-rows":"group-rows", u.qualify(family),
        [carrier](Operation *op, Attribute a, const TargetProfile &, RewriterBase &) {
          auto d=cast<DictionaryAttr>(a);
          op->setAttr("cpu.row_groups",d.get("rows"));op->setAttr("cpu.walk",d.get("walk"));op->setAttr("cpu.tile",d.get("tile"));
          op->removeAttr(carrier);
          return llvm::Error::success();
        }},
       COMPILERINFRA_SOURCE});
}
void rowTasks(Unit &u){registerPartition(u,false);registerPartition(u,true);}
} // namespace CI_UNIT_NAMESPACE
