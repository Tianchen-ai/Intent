#include "intentdsl/levels/execution/cpu/ir/Task.h"
using namespace mlir;
using namespace compilerinfra;
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
    auto rows = dyn_cast<DenseI64ArrayAttr>(a);
    auto *machine = p.services.get<intentdsl::cpu::RowMachine>();
    if (!machine || !rows || rows.empty() ||
        !llvm::all_of(rows.asArrayRef(), [&](auto r) {
          return llvm::is_contained(machine->rows, r);
        }))
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
                    return Attribute(DenseI64ArrayAttr::get(op->getContext(),
                                                            machine->rows));
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
