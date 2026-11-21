#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::cpu {
ArrayAttr RowMachine::configurations(MLIRContext *ctx) const {
  Builder b(ctx);
  SmallVector<Attribute> values;
  for (auto [r, c] : rows)
    values.push_back(b.getDictionaryAttr(
        {b.getNamedAttr("grain", b.getI64IntegerAttr(r)),
         b.getNamedAttr("columns", b.getI64IntegerAttr(c))}));
  return b.getArrayAttr(values);
}
bool RowMachine::accepts(ArrayAttr values) const {
  if (!values || values.empty())
    return false;
  auto allowed = configurations(values.getContext());
  llvm::SmallDenseSet<Attribute> seen;
  return llvm::all_of(values, [&](Attribute row) {
    return llvm::is_contained(allowed, row) && seen.insert(row).second;
  });
}
} // namespace intentdsl::cpu
namespace CI_UNIT_NAMESPACE {
void rowTasks(Unit &u) {
  DecisionFamily partition;
  partition.name = "partition";
  partition.applies = [](Operation *op) {
    return op->getName().getStringRef() == "cpu.parallel";
  };
  partition.read = [](Operation *op) { return op->getAttr("cpu.partition"); };
  partition.write = [](Operation *op, Attribute a) {
    op->setAttr("cpu.partition", a);
    return llvm::Error::success();
  };
  partition.verify = [](Operation *, Attribute a,
                        const TargetProfile &p) -> llvm::Error {
    auto *machine = p.services.get<intentdsl::cpu::RowMachine>();
    if (!machine || !machine->accepts(dyn_cast<ArrayAttr>(a)))
      return llvm::createStringError("illegal CPU work partition");
    return llvm::Error::success();
  };
  u.families.push_back(
      {{partition, "cpu@formed", "cpu@formed"}, COMPILERINFRA_SOURCE});
  Strategy select{"row-reduction", u.qualify("partition"),
                  [](Operation *op, const TargetProfile &p,
                     AnalysisManager &) -> llvm::Expected<Attribute> {
                    auto *machine =
                        p.services.get<intentdsl::cpu::RowMachine>();
                    if (!machine)
                      return llvm::createStringError("missing CPU capacity");
                    bool reduction = false;
                    op->walk([&](Operation *i) {
                      reduction |= i->getName().stripDialect() == "sum" ||
                                   i->getName().stripDialect() == "max_num";
                    });
                    if (!reduction)
                      return Attribute{};
                    return Attribute(machine->configurations(op->getContext()));
                  }};
  u.strategies.push_back({select, COMPILERINFRA_SOURCE});
  u.realizers.push_back(
      {{"group-rows", u.qualify("partition"),
        [](Operation *op, Attribute a, const TargetProfile &, RewriterBase &) {
          op->setAttr("cpu.row_groups", a);
          op->removeAttr("cpu.partition");
          return llvm::Error::success();
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
