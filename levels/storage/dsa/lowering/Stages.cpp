#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "../decisions/Plan.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
namespace intentdsl::dsa {
namespace {
Operation *inst(OpBuilder &b, Location l, StringRef name, ValueRange args = {},
                TypeRange types = {}, ArrayRef<NamedAttribute> attrs = {}) {
  OperationState s(l, name);
  s.addOperands(args);
  s.addTypes(types);
  s.addAttributes(attrs);
  return b.create(s);
}
void schedule(Block *entry, bool stream, int64_t depth,bool persistent, OpBuilder &b,
              Location loc) {
  b.setInsertionPointToEnd(entry);
  auto idx = b.getIndexType();
  auto range = inst(
      b, loc, "dsa.range", {}, {idx, idx, idx},
      b.getNamedAttr("source", b.getStringAttr(stream ? "tile_extent"
                                                      : "participant_stride")));
  auto init = inst(b, loc, "dsa.slot_zero", {}, idx)->getResult(0);
  auto lo = range->getResult(0), end = range->getResult(1),
       step = range->getResult(2);
  if(persistent)inst(b,loc,"dsa.load_shared",{lo,end});
  if (depth == 2)
    inst(b, loc, "dsa.issue_if_in_bounds", {lo, end, init});
  OperationState loop(loc, "dsa.iterate");
  loop.addOperands({lo, end, step, init});
  loop.addRegion();
  auto *body = b.createBlock(&b.create(loop)->getRegion(0));
  Value pos = body->addArgument(idx, loc), slot = body->addArgument(idx, loc),
        nextSlot = slot;
  if (depth == 2) {
    inst(b, loc, "dsa.wait_slot", slot);
    auto next = inst(b, loc, "dsa.next", {pos, step}, idx)->getResult(0);
    nextSlot = inst(b, loc, "dsa.other_slot", slot, idx)->getResult(0);
    inst(b, loc, "dsa.issue_if_in_bounds", {next, end, nextSlot});
  } else
    inst(b, loc, "dsa.issue_if_in_bounds", {pos, end, slot});
  inst(b, loc, "dsa.compute_members", {pos, slot});
  inst(b, loc, "dsa.advance", nextSlot);
  b.setInsertionPointToEnd(entry);
  inst(b, loc, "dsa.end");
}
LogicalResult phase(Operation *anchor, Block &source, bool stream,
                    int64_t depth, RewriterBase &b) {
  b.setInsertionPoint(anchor);
  auto loc = anchor->getLoc();
  auto inputs = anchor->getOperands();
  OperationState ps(loc, "dsa.phase");
  ps.addAttribute("axis", b.getStringAttr(stream ? "columns" : "instances"));
  ps.addAttribute("slots", b.getI32IntegerAttr(depth));
  ps.addAttribute("lookahead", b.getI32IntegerAttr(depth - 1));
  ps.addAttribute("visibility", b.getStringAttr("participant_private"));
  if (stream) {
    ps.addOperands(inputs.drop_front());
    ps.addTypes(anchor->getResultTypes());
  }
  ps.addRegion();
  ps.addRegion();
  ps.addRegion();
  auto *p = b.create(ps);
  auto *fetch = b.createBlock(&p->getRegion(0));
  auto *calc = b.createBlock(&p->getRegion(1));
  auto fn=anchor->getParentOfType<func::FuncOp>();
  bool persistent=fn->getAttrOfType<DictionaryAttr>("dsa.pending").getAs<BoolAttr>("persistent").getValue();
  schedule(b.createBlock(&p->getRegion(2)), stream, depth,persistent, b, loc);
  Operation *active = nullptr;
  for (auto &op : source)
    if (op.getName().getStringRef() == "block.active")
      active = &op;
  if (!active)
    return p->emitError("missing block validity domain");
  auto window = [&](Block *where) {
    b.setInsertionPointToEnd(where);
    return inst(b, loc, "dsa.window", active->getOperand(0),
                {active->getOperand(1).getType(),
                 active->getOperand(2).getType(),
                 active->getResult(0).getType()});
  };
  auto *readWindow = window(fetch), *writeWindow = window(calc);
  auto coords=[&](Operation *op,Operation *w){
    b.setInsertionPointAfter(w);
    SmallVector<Value> axes;
    auto domain=w->getOperand(0);
    auto rank=cast<MemRefType>(domain.getType()).getRank();
    auto mapping=op->getAttrOfType<DenseI64ArrayAttr>("block.member_axes");
    if(auto owner=fn->getAttrOfType<IntegerAttr>("kir.owner_axis")){
      for(int i=0;i<rank;++i)
        if(i==owner.getInt())axes.push_back(w->getResult(0));
        else axes.push_back(inst(b,loc,"dsa.slab_axis",{w->getResult(1),domain},w->getResult(1).getType(),
                                 b.getNamedAttr("axis",b.getI64IntegerAttr(i)))->getResult(0));
    }else if(mapping){
      for(auto axis:mapping.asArrayRef()){
        if(axis==rank-1)axes.push_back(w->getResult(1));
        else if(rank==2 && axis==0)axes.push_back(w->getResult(0));
        else axes.push_back(inst(b,loc,"dsa.member_axis",{w->getResult(0),domain},w->getResult(0).getType(),
                    b.getNamedAttr("axis",b.getI64IntegerAttr(axis)))->getResult(0));
      }
    }else{
      for(auto v:op->getOperands().slice(1,2)){
        if(v==active->getOperand(1))axes.push_back(w->getResult(0));
        else if(v==active->getOperand(2))axes.push_back(w->getResult(1));
      }
      if(cast<MemRefType>(op->getOperand(0).getType()).getRank()==1 && axes.size()==2)axes.erase(axes.begin());
    }
    b.setInsertionPointToEnd(w->getBlock());
    return axes;
  };
  SmallVector<Value> reads;
  IRMapping map;
  SmallVector<Operation *> old;
  for (auto &op : source)
    if (&op != p)
      old.push_back(&op);
  for (Operation *op : old) {
    if (op->getName().getStringRef() != "block.load")
      continue;
    auto view = op->getOperand(0);
    b.setInsertionPointToEnd(fetch);
    auto axes = coords(op, readWindow);
    if (axes.size() != size_t(cast<MemRefType>(op->getOperand(0).getType()).getRank()) || op->getOperand(3) != active->getResult(0))
      return op->emitError("mismatched access domain");
    SmallVector<Value> address{view, readWindow->getResult(2)};
    address.append(axes);
    auto *copy = inst(b, loc, "dsa.copy_in", address, op->getResultTypes());
    if(persistent && cast<MemRefType>(view.getType()).getRank()==1)copy->setAttr("dsa.shared",b.getUnitAttr());
    reads.push_back(copy->getResult(0));
    b.setInsertionPointToEnd(calc);
    auto pending = calc->addArgument(copy->getResult(0).getType(), loc);
    map.map(
        op->getResult(0),
        inst(b, loc, "dsa.await", pending, pending.getType())->getResult(0));
  }
  b.setInsertionPointToEnd(fetch);
  inst(b, loc, "dsa.issued", reads);
  b.setInsertionPointToEnd(calc);
  if (stream && p->getNumResults())
    map.map(source.getArgument(1),
            calc->addArgument(p->getResult(0).getType(), loc));
  SmallVector<Value> yields, completions;
  for (auto *op : old) {
    auto code = op->getName().stripDialect();
    if (code == "pid" || code == "rows" || code == "columns" ||
        code == "active" || code == "zero" || code == "load" ||
        code == "return")
      continue;
    auto val = [&](unsigned i) {
      return map.lookupOrDefault(op->getOperand(i));
    };
    if(code=="owner_store") {
      auto done=inst(b,loc,"dsa.owner_write",{val(0),writeWindow->getResult(0),val(2)},b.getI1Type());
      completions.push_back(done->getResult(0));
    } else if(code=="owner_load") {
      map.map(op->getResult(0),inst(b,loc,"dsa.owner_read",{val(0),writeWindow->getResult(0)},op->getResultTypes())->getResult(0));
    } else if(code=="summary_reduce"){
      map.map(op->getResult(0),inst(b,loc,"block.summary_reduce",{val(0),writeWindow->getResult(2)},op->getResultTypes())->getResult(0));
    } else if (code == "linear_index") {
      auto *index = inst(b, loc, "block.linear_index",
                         {active->getOperand(0), writeWindow->getResult(0),
                          writeWindow->getResult(1)},
                         op->getResultTypes());
      map.map(op->getResult(0), index->getResult(0));
    } else if (code == "reduce") {
      auto r = inst(b, loc, "dsa.reduce", {val(0), writeWindow->getResult(2)},
                    op->getResultTypes(), op->getAttrs());
      map.map(op->getResult(0), r->getResult(0));
    } else if (code == "store") {
      auto axes = coords(op, writeWindow);
      if (axes.size() != size_t(cast<MemRefType>(op->getOperand(0).getType()).getRank()) || op->getOperand(3) != active->getResult(0))
        return op->emitError("inconsistent write domain");
      SmallVector<Value> address{val(0),val(4),writeWindow->getResult(2)};
      address.append(axes);
      auto *store =
          inst(b, loc, "dsa.write_back",
               address,
               b.getI1Type(), b.getNamedAttr("async", b.getBoolAttr(false)));
      completions.push_back(store->getResult(0));
    } else if (code == "yield") {
      for (auto v : op->getOperands())
        yields.push_back(map.lookupOrDefault(v));
    } else
      b.clone(*op, map);
  }
  auto done =
      inst(b, loc, "dsa.drain", completions, b.getI1Type())->getResult(0);
  yields.insert(yields.begin(), done);
  inst(b, loc, "dsa.complete", yields);
  if (stream)
    b.replaceOp(anchor, p->getResults());
  else
    for (auto *op : llvm::reverse(old))
      if (op->getName().getStringRef() != "func.return")
        b.eraseOp(op);
  return success();
}
} // namespace
llvm::Error stageTransfers(Operation *op, Attribute a,
                           const compilerinfra::TargetProfile &,
                           RewriterBase &b) {
  auto fn = cast<func::FuncOp>(op);
  auto decision = cast<DictionaryAttr>(a);
  // Load the registered dialect also when resuming from a Block checkpoint.
  fn.getContext()->getOrLoadDialect("dsa");
  if (fn->getAttrOfType<StringAttr>("block.topology") == "resident") {
    if (failed(phase(&fn.front().front(), fn.front(), false,
                     decision.getAs<IntegerAttr>("depth").getInt(), b)))
      return llvm::createStringError("unable to form resident phase");
  } else {
    SmallVector<Operation *> loops;
    fn.walk([&](Operation *o) {
      if (o->getName().getStringRef() == "block.tiles")
        loops.push_back(o);
    });
    for (auto *l : loops)
      if (failed(phase(l, l->getRegion(0).front(), true,
                       decision.getAs<IntegerAttr>("depth").getInt(), b)))
        return llvm::createStringError("unable to form stream phase");
    SmallVector<Value> writes;
    auto domain=fn.getArgument(fn->getAttrOfType<IntegerAttr>("kir.domain_arg").getInt());
    for(auto &item:llvm::make_early_inc_range(fn.front())){
      auto code=item.getName().getStringRef();
      b.setInsertionPoint(&item);
      if(code=="block.rows" && !item.use_empty()){
        auto *rows=inst(b,fn.getLoc(),"dsa.owner_rows",domain,item.getResultTypes());
        b.replaceOp(&item,rows->getResults());
      } else if(code=="block.owner_load"){
        auto *load=inst(b,item.getLoc(),"dsa.owner_read",item.getOperands(),item.getResultTypes());
        b.replaceOp(&item,load->getResults());
      } else if(code=="block.owner_store"){
        auto *write=inst(b,item.getLoc(),"dsa.owner_write",item.getOperands(),b.getI1Type());
        writes.push_back(write->getResult(0));
        b.eraseOp(&item);
      }
    }
    if(!writes.empty()){
      b.setInsertionPoint(fn.front().getTerminator());
      inst(b,fn.getLoc(),"dsa.drain",writes,b.getI1Type());
    }
    for (auto &item : llvm::make_early_inc_range(llvm::reverse(fn.front())))
      if (item.use_empty() && (item.getName().getStringRef() == "block.rows" ||
                               item.getName().getStringRef() == "block.pid"))
        b.eraseOp(&item);
    b.setInsertionPointToStart(&fn.front());
    auto index = b.getIndexType();
    auto *range =
        inst(b, fn.getLoc(), "dsa.range", {}, {index, index, index},
             b.getNamedAttr("source", b.getStringAttr("participant_stride")));
    OperationState gs(fn.getLoc(), "dsa.task_groups");
    gs.addOperands(range->getResults());
    gs.addRegion();
    auto *groups = b.create(gs);
    auto *group = b.createBlock(&groups->getRegion(0));
    for (auto &op : llvm::make_early_inc_range(fn.front().without_terminator()))
      if (&op != groups && &op != range)
        op.moveBefore(group, group->end());
    b.setInsertionPointToEnd(group);
    inst(b, fn.getLoc(), "dsa.end");
  }
  fn->setAttr("dsa.layout", decision);
  fn->removeAttr("dsa.pending");
  return llvm::Error::success();
}
} // namespace intentdsl::dsa
namespace intentdsl::dsa { void reuseFragments(mlir::OpPassManager &); }
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
void dialect(DialectRegistry &);
void supply(Unit &);
void supplyCost(Unit &);
void bind(Unit &u) {
  supplyCost(u);
  u.dialects = dialect;
  supply(u);
  for (bool ready : {false, true})
    u.boundaries.push_back({{ready ? "supplied" : "formed",
                             [ready](OpPassManager &pm, const TargetProfile &) {
                               if (!ready) intentdsl::dsa::reuseFragments(pm);
                               return llvm::Error::success();
                             },
                             [ready](ModuleOp m, const TargetProfile &p) {
                               return intentdsl::dsa::verifyProgram(m, p,
                                                                    ready);
                             }},
                            COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
