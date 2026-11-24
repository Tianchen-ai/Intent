#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
using namespace mlir;
namespace intentdsl::block {
bool canGroupReads(Operation *o, const compilerinfra::TargetProfile &p,
                   unsigned width) {
  auto machine = p.services.get<OwnerMachine>();
  if (!machine || machine->parallelMembers <= 1 ||
      int(width) > machine->maxGroupReads)
    return false;
  auto fn = o->getParentOfType<func::FuncOp>();
  auto abi = llvm::cantFail(
      llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi")));
  if (llvm::any_of(*abi.getAsArray(), [](auto &s) {
        return s.getAsObject()->getString("direction") == "InOut";
      }))
    return false;
  auto loops = o->getRegion(0).front().getOps<scf::ForOp>();
  if (!llvm::hasSingleElement(loops))
    return false;
  auto loop = *loops.begin();
  auto d = loop.getUpperBound().getDefiningOp<memref::DimOp>();
  if (!d || !d.getConstantIndex() ||
      !matchPattern(loop.getLowerBound(), m_Zero()) ||
      !matchPattern(loop.getStep(), m_One()) ||
      loop.getNumRegionIterArgs() == 0 ||
      llvm::any_of(*loop.getBody(),
                   [](Operation &o) { return o.getNumRegions() != 0; }))
    return false;
  auto n = (*p.services.get<CallFacts>()->shapeOf(
      d.getSource()))[*d.getConstantIndex()]
               .getAsInteger();
  return n && *n >= width && *n % width == 0;
}
void groupReads(func::FuncOp fn, unsigned width) {
  auto &space = fn.front().front();
  auto old = *space.getRegion(0).front().getOps<scf::ForOp>().begin();
  auto loc = old.getLoc();
  OpBuilder b(old);
  auto step = b.create<arith::ConstantIndexOp>(loc, width);
  auto loop = b.create<scf::ForOp>(
      loc, old.getLowerBound(), old.getUpperBound(), step, old.getInitArgs());
  SmallVector<Value> state(loop.getRegionIterArgs());
  SmallVector<IRMapping> maps(width);
  b.setInsertionPointToStart(loop.getBody());
  // Mark the address closure of reads which do not depend on a carry.
  llvm::SmallPtrSet<Operation *, 32> prefetch;
  auto independent = [&](auto &self, Value v) -> bool {
    if (old.isDefinedOutsideOfLoop(v) || v == old.getInductionVar())
      return true;
    auto *def = v.getDefiningOp();
    if (!def || def->getNumRegions() || !isMemoryEffectFree(def))
      return false;
    return llvm::all_of(def->getOperands(),
                        [&](Value x) { return self(self, x); });
  };
  auto mark = [&](auto &self, Operation *v) -> void {
    if (!prefetch.insert(v).second)
      return;
    for (auto a : v->getOperands())
      if (a.getDefiningOp() && a.getDefiningOp()->getBlock() == old.getBody())
        self(self, a.getDefiningOp());
  };
  for (auto &op : old.getBody()->without_terminator())
    if ((isa<memref::LoadOp>(op) ||
         op.getName().stripDialect() == "slice_read") &&
        llvm::all_of(op.getOperands(),
                     [&](Value v) { return independent(independent, v); }))
      mark(mark, &op);
  for (unsigned i = 0; i < width; ++i) {
    Value iv = loop.getInductionVar();
    if (i)
      iv = b.create<arith::AddIOp>(loc, iv,
                                   b.create<arith::ConstantIndexOp>(loc, i));
    maps[i].map(old.getInductionVar(), iv);
    for (auto &op : old.getBody()->without_terminator())
      if (prefetch.contains(&op))
        b.clone(op, maps[i]);
  }
  for (auto &map : maps) {
    map.map(old.getRegionIterArgs(), state);
    for (auto &op : old.getBody()->without_terminator())
      if (!prefetch.contains(&op))
        b.clone(op, map);
    state.clear();
    for (auto v : old.getBody()->getTerminator()->getOperands())
      state.push_back(map.lookupOrDefault(v));
  }
  b.create<scf::YieldOp>(loc, state);
  old.replaceAllUsesWith(loop.getResults());
  old.erase();
}
} // namespace intentdsl::block
