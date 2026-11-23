#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "intentdsl/levels/execution/cpu/ir/RepeatedScore.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
using namespace mlir;
namespace intentdsl::cpu {
static bool sameInterval(Value v, BlockArgument head, Value projected) {
  if (v == projected)
    return true;
  if (v == head)
    return false;
  if (auto a = dyn_cast<BlockArgument>(v)) {
    if (auto loop = dyn_cast<scf::ForOp>(a.getOwner()->getParentOp()))
      return v == loop.getInductionVar() &&
             sameInterval(loop.getLowerBound(), head, projected);
    return true;
  }
  auto *p = v.getDefiningOp();
  if (auto branch = dyn_cast<scf::IfOp>(p))
    return branch.getNumResults() == 1 &&
           sameInterval(branch.getCondition(), head, projected) &&
           sameInterval(branch.thenYield().getOperand(0), head, projected) &&
           sameInterval(branch.elseYield().getOperand(0), head, projected);
  return (isa<memref::LoadOp, memref::DimOp>(p) ||
          p->getName().stripDialect() == "slice_read" ||
          p->getName().getDialectNamespace() == "arith") &&
         llvm::all_of(p->getOperands(), [&](Value x) {
           return sameInterval(x, head, projected);
         });
}
bool invariantOwnerValue(Value v, Operation &space, const OwnerReuse &r) {
  return sameInterval(v, space.getRegion(0).front().getArgument(r.axis),
                      r.quotient);
}
std::optional<OwnerReuse> sharedInterval(Operation *space,
                                         const CallFacts &facts) {
  auto &body = space->getRegion(0).front();
  auto axes = space->getAttrOfType<DenseI64ArrayAttr>("axes");
  // Interleaving is legal only for read-only folds followed by publications.
  // Any update of a captured view inside a fold keeps its original order.
  auto unsafe = body.walk([&](Operation *op) {
    if (op->getName().stripDialect() == "slice_write" ||
        isa<memref::StoreOp>(op))
      return op->getBlock() == &body ? WalkResult::advance()
                                     : WalkResult::interrupt();
    if (isa<scf::ForOp>(op) && op->getBlock() != &body)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (unsafe.wasInterrupted())
    return {};
  for (auto d : body.getOps<arith::FloorDivSIOp>()) {
    auto h = dyn_cast<BlockArgument>(d.getLhs());
    auto group = d.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    if (!h || h.getOwner() != &body || !group || group.value() < 2 ||
        h.getArgNumber() + 1 >= body.getNumArguments())
      continue;
    double bytes = 0;
    for (auto loop : body.getOps<scf::ForOp>()) {
      if (!sameInterval(loop.getLowerBound(), h, d) ||
          !sameInterval(loop.getUpperBound(), h, d) ||
          !sameInterval(loop.getStep(), h, d)) {
        bytes = 0;
        break;
      }
      for (auto &read : loop.getBody()->without_terminator()) {
        if (read.getName().stripDialect() != "slice_read")
          continue;
        auto cs = read.getOperands().drop_front();
        if (!llvm::is_contained(cs, d.getResult()) ||
            !llvm::all_of(cs, [&](Value v) { return sameInterval(v, h, d); }))
          continue;
        auto *s = facts.shapeOf(read.getOperand(0));
        if (!s)
          continue;
        bytes += (*s)[read.getAttrOfType<IntegerAttr>("axis").getInt()]
                     .getAsInteger()
                     .value_or(0) *
                 cast<ShapedType>(read.getResult(0).getType())
                     .getElementTypeBitWidth() /
                 8.;
      }
    }
    auto axis = h.getArgNumber();
    auto shape = facts.shapeOf(space->getOperand(axis));
    if (bytes > 0 && shape) {
      auto extent = (*shape)[axes[axis]].getAsInteger();
      if (extent && *extent > 0)
        return OwnerReuse{axis, group.value(), *extent, bytes, d};
    }
  }
  return {};
}
llvm::SmallVector<unsigned> ownerRadices(Operation &space) {
  auto group = space.getAttrOfType<IntegerAttr>("cpu.innermost_owner");
  llvm::SmallVector<unsigned> result;
  if (group)
    result.push_back(group.getInt());
  for (unsigned i = space.getNumOperands(); i-- > 0;)
    if (!group || group.getInt() != i)
      result.push_back(i);
  return result;
}
} // namespace intentdsl::cpu
namespace CI_UNIT_NAMESPACE {
void sharedOwnerTraversal(compilerinfra::Unit &u) {
  using namespace compilerinfra;
  Strategy s;
  s.name = "shared-intervals";
  s.family = u.qualify("ordered-owners");
  s.enumerate = [](Operation *op, const TargetProfile &p, AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    std::vector<StructuralCandidate> cs;
    auto reuse = intentdsl::cpu::sharedInterval(
        op, *p.services.get<intentdsl::CallFacts>());
    auto maximum = p.services.get<intentdsl::cpu::RowMachine>()->packedOwners;
    if (!reuse || !maximum)
      return cs;
    Builder b(op->getContext());
    bool repeat = bool(intentdsl::cpu::repeatedScore(*op));
    for (int grain : {8, 32, 128, 256})
      for (unsigned batch : {2u, 4u, 8u, 16u}) {
        if (batch > maximum || grain % batch || reuse->group % batch ||
            reuse->extent % batch)
          continue;
        auto d = b.getDictionaryAttr(
            {b.getNamedAttr("grain", b.getI64IntegerAttr(grain)),
             b.getNamedAttr("inner", b.getI64IntegerAttr(reuse->axis)),
             b.getNamedAttr("batch", b.getI64IntegerAttr(batch))});
        cs.push_back(
            {"shared-" + std::to_string(grain) + "x" + std::to_string(batch),
             "shared-interval-owners",
             d,
             {"exclusive-output"}});
        if (repeat) {
          NamedAttrList row(d);
          row.set("window",
                  b.getI64IntegerAttr(intentdsl::cpu::batchScoreSlots / batch));
          cs.push_back({cs.back().id + "-scores",
                        "shared-interval-owners",
                        row.getDictionary(b.getContext()),
                        {"exclusive-output"}});
        }
      }
    return cs;
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
