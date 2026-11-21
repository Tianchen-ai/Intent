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
  auto *p = b.create(ps);
  auto *fetch = b.createBlock(&p->getRegion(0));
  auto *calc = b.createBlock(&p->getRegion(1));
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
    auto projection = intentdsl::block::projection(view.getType());
    if (failed(projection))
      return op->emitError("cannot stage unprojected view");
    b.setInsertionPointToEnd(fetch);
    auto *copy = inst(
        b, loc, "dsa.copy_in", view, op->getResultTypes(),
        b.getNamedAttr("matrix",
                       b.getBoolAttr(*projection ==
                                     intentdsl::block::RowProjection::Matrix)));
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
  SmallVector<Value> yields;
  for (auto *op : old) {
    auto code = op->getName().stripDialect();
    if (code == "pid" || code == "rows" || code == "columns" ||
        code == "active" || code == "zero" || code == "load" ||
        code == "return")
      continue;
    auto val = [&](unsigned i) {
      return map.lookupOrDefault(op->getOperand(i));
    };
    if (code == "reduce") {
      auto r = inst(b, loc, "dsa.reduce", val(0), op->getResultTypes(),
                    op->getAttrs());
      map.map(op->getResult(0), r->getResult(0));
    } else if (code == "store")
      inst(b, loc, "dsa.write_back", {val(0), val(4)});
    else if (code == "yield") {
      for (auto v : op->getOperands())
        yields.push_back(map.lookupOrDefault(v));
    } else
      b.clone(*op, map);
  }
  inst(b, loc, "dsa.drain");
  inst(b, loc, "dsa.yield", yields);
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
    for (auto &item : llvm::make_early_inc_range(llvm::reverse(fn.front())))
      if (item.use_empty() && (item.getName().getStringRef() == "block.rows" ||
                               item.getName().getStringRef() == "block.pid"))
        b.eraseOp(&item);
  }
  fn->setAttr("dsa.layout", decision);
  fn->removeAttr("dsa.pending");
  return llvm::Error::success();
}
} // namespace intentdsl::dsa
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
void dialect(DialectRegistry &);
void supply(Unit &);
void bind(Unit &u) {
  u.dialects = dialect;
  supply(u);
  for (bool ready : {false, true})
    u.boundaries.push_back({{ready ? "supplied" : "formed",
                             [](OpPassManager &, const TargetProfile &) {
                               return llvm::Error::success();
                             },
                             [ready](ModuleOp m, const TargetProfile &p) {
                               return intentdsl::dsa::verifyProgram(m, p,
                                                                    ready);
                             }},
                            COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
