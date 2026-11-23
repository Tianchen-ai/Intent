#include "intentdsl/levels/execution/block/ir/SharedReads.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
using namespace mlir;
namespace intentdsl::block {
static bool invariant(Value v, BlockArgument owner, int64_t group) {
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()))
      return arg == loop.getInductionVar() &&
             invariant(loop.getLowerBound(), owner, group);
    return arg != owner;
  }
  auto *op = v.getDefiningOp();
  if (auto div = dyn_cast<arith::FloorDivSIOp>(op)) {
    auto by = div.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    if (div.getLhs() == owner && by && by.value() == group)
      return true;
  }
  if (isa<memref::DimOp>(op))
    return true;
  if (!isa<memref::LoadOp>(op) &&
      op->getName().getDialectNamespace() != "arith")
    return false;
  return llvm::all_of(op->getOperands(),
                      [&](Value u) { return invariant(u, owner, group); });
}
std::optional<SharedReads> sharedReads(Operation *op, const CallFacts &facts) {
  auto &body = op->getRegion(0).front();
  auto axes = op->getAttrOfType<DenseI64ArrayAttr>("axes");
  for (auto div : body.getOps<arith::FloorDivSIOp>()) {
    auto owner = dyn_cast<BlockArgument>(div.getLhs());
    auto factor = div.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    if (!owner || owner.getOwner() != &body || !factor || factor.value() < 2)
      continue;
    unsigned axis = owner.getArgNumber();
    auto *shape = facts.shapeOf(op->getOperand(axis));
    auto extent = shape ? (*shape)[axes[axis]].getAsInteger() : std::nullopt;
    if (!extent || *extent <= 0 || *extent % factor.value())
      continue;
    SharedReads reads{axis, factor.value(), {}};
    op->walk([&](Operation *s) {
      if (s->getName().stripDialect() != "slice_read" ||
          !s->getParentOfType<scf::ForOp>())
        return;
      for (auto *p = s->getParentOp(); p != op; p = p->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(p)) {
          if (!invariant(loop.getLowerBound(), owner, reads.group) ||
              !invariant(loop.getUpperBound(), owner, reads.group))
            return;
        } else if (auto branch = dyn_cast<scf::IfOp>(p)) {
          if (!invariant(branch.getCondition(), owner, reads.group))
            return;
        } else
          return;
      }
      auto coords = s->getOperands().drop_front();
      if (llvm::is_contained(coords, div.getResult()) &&
          llvm::all_of(coords, [&](Value v) {
            return invariant(v, owner, reads.group);
          }))
        reads.slices.push_back(s);
    });
    if (!reads.slices.empty())
      return reads;
  }
  return std::nullopt;
}
bool legalSharedReads(Operation *o, const compilerinfra::TargetProfile &p,
                      DictionaryAttr d) {
  auto machine = p.services.get<OwnerMachine>();
  auto facts = p.services.get<CallFacts>();
  auto w = d.getAs<IntegerAttr>("width"), ax = d.getAs<IntegerAttr>("axis");
  if (!machine || machine->parallelMembers <= 1 || !facts || !w || !ax ||
      w.getInt() < 2 || w.getInt() > machine->parallelMembers ||
      !llvm::isPowerOf2_64(w.getInt()))
    return false;
  auto r = sharedReads(o, *facts);
  return r && ax.getInt() == r->axis && r->group % w.getInt() == 0;
}
LogicalResult verifySharedReads(Operation *o, const CallFacts &facts) {
  auto ax = o->getAttrOfType<IntegerAttr>("block.group_axis");
  SmallVector<Operation *> marked;
  o->walk([&](Operation *s) {
    if (s->hasAttr("block.broadcast_read"))
      marked.push_back(s);
  });
  if (!ax && marked.empty())
    return success();
  auto r = sharedReads(o, facts);
  auto w = o->getAttrOfType<IntegerAttr>("block.owner_tile");
  if (!ax || !w || w.getInt() < 2 || !r || ax.getInt() != r->axis ||
      r->group % w.getInt() || r->slices != marked)
    return o->emitError(
        "owner broadcasts require identical addresses within each owner group");
  return success();
}
} // namespace intentdsl::block
