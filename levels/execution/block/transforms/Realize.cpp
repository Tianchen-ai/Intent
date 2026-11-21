#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
namespace {
class RowTile {
public:
  RowTile(Operation *anchor, DictionaryAttr choice, RewriterBase &rw)
      : anchor(anchor), rw(rw), loc(anchor->getLoc()),
        domain(anchor->getOperand(0)),
        resident(choice.getAs<StringAttr>("topology") == "resident") {
    auto fn = anchor->getParentOfType<func::FuncOp>();
    fn->setAttr("block.domain_arg", rw.getI64IntegerAttr(cast<BlockArgument>(domain).getArgNumber()));
    fn->setAttr("block.guard",choice.get("guard"));
    fn->setAttr("block.grid", rw.getStrArrayAttr({"ceildiv", "M", "rows"}));
    fn->setAttr("block.configs", choice.get("rows"));
    fn->setAttr("block.topology", choice.get("topology"));
  }
  void build() {
    rw.setInsertionPoint(anchor);
    auto pid = make("pid", {}, rw.getIndexType(), rw.getNamedAttr("axis",rw.getI32IntegerAttr(0)))->getResult(0);
    rows = make("rows", pid, vector(rw.getIndexType(), 1), rw.getNamedAttr("step",rw.getStringAttr("rows")))->getResult(0);
    auto &body = anchor->getRegion(0).front();
    if (resident) {
      tile(make("zero", {}, rw.getIndexType())->getResult(0));
      for (auto &op : body.without_terminator()) {
        if (isReduction(&op))
          fixed.map(op.getResult(0), reduce(&op));
        else if (op.getName().stripDialect() == "write")
          store(&op);
      }
    } else {
      for (auto &op : body.without_terminator()) {
        if (isReduction(&op))
          sweep(&op);
        else if (op.getName().stripDialect() == "write")
          sweep(&op);
      }
    }
    rw.eraseOp(anchor);
  }

private:
  Operation *anchor;
  RewriterBase &rw;
  Location loc;
  Value domain, rows, cols, active;
  bool resident;
  IRMapping fixed, local;
  Type vector(Type element, unsigned rank) {
    return RankedTensorType::get(
        SmallVector<int64_t>(rank, ShapedType::kDynamic), element);
  }
  Operation *make(StringRef code, ValueRange args, TypeRange results,
                  ArrayRef<NamedAttribute> attrs = {}) {
    OperationState state(loc, ("block." + code).str());
    state.addOperands(args);
    state.addTypes(results);
    state.addAttributes(attrs);
    return rw.create(state);
  }
  bool isReduction(Operation *op) {
    auto n = op->getName().stripDialect();
    return n == "sum" || n == "max_num";
  }
  void tile(Value i) {
    cols = make("columns", i, vector(rw.getIndexType(), 1))->getResult(0);
    active = make("active", {domain, rows, cols}, vector(rw.getI1Type(), 2))
                 ->getResult(0);
    local.clear();
  }
  Value expr(Value v) {
    if (auto known = fixed.lookupOrNull(v))
      return known;
    if (auto known = local.lookupOrNull(v))
      return known;
    auto *op = v.getDefiningOp();
    if (!op)
      return v;
    auto n = op->getName().stripDialect();
    if (n == "read") {
      auto view = op->getOperand(0);
      auto elem = cast<MemRefType>(view.getType()).getElementType();
      auto x = make("load", {view, rows, cols, active}, vector(elem, 2))
                   ->getResult(0);
      local.map(v, x);
      return x;
    }
    SmallVector<Value> args;
    unsigned rank = 0;
    for (Value operand : op->getOperands()) {
      auto input = expr(operand);
      args.push_back(input);
      if (auto shaped = dyn_cast<RankedTensorType>(input.getType()))
        rank = std::max(rank, unsigned(shaped.getRank()));
    }
    auto element = v.getType();
    if (auto shaped = dyn_cast<RankedTensorType>(element))
      element = shaped.getElementType();
    auto x =
        make(n, args, rank ? vector(element, rank) : element)->getResult(0);
    local.map(v, x);
    return x;
  }
  NamedAttribute algebra(Operation *op) {
    return rw.getNamedAttr("algebra",
                           rw.getStringAttr(op->getName().stripDialect()));
  }
  Value reduce(Operation *op) {
    return make("reduce", {expr(op->getOperand(0)), active},
                vector(op->getResult(0).getType(), 1), algebra(op))
        ->getResult(0);
  }
  void store(Operation *op) {
    make("store",
         {op->getOperand(0), rows, cols, active, expr(op->getOperand(2))}, {});
  }
  void sweep(Operation *op) {
    bool accum = isReduction(op);
    OperationState state(loc, "block.tiles");
    state.addOperands(domain);
    if (accum) {
      auto t = vector(op->getResult(0).getType(), 1);
      state.addOperands(make("identity", {}, t, algebra(op))->getResult(0));
      state.addTypes(t);
    }
    state.addRegion();
    auto *loop = rw.create(state);
    {
      OpBuilder::InsertionGuard guard(rw);
      auto *bb = rw.createBlock(&loop->getRegion(0));
      tile(bb->addArgument(rw.getIndexType(), loc));
      if (accum) {
        auto prev = bb->addArgument(loop->getResult(0).getType(), loc);
        auto combined =
            make("combine", {prev, reduce(op)}, prev.getType(), algebra(op));
        make("yield", combined->getResults(), {});
      } else {
        store(op);
        make("yield", {}, {});
      }
    }
    if (accum)
      fixed.map(op->getResult(0), loop->getResult(0));
  }
};
} // namespace
void registerRealizer(Unit &unit) {
  Realizer r;
  r.name = "tile-rows";
  r.family = unit.qualify("structure");
  r.apply = [](Operation *op, Attribute a, const TargetProfile &,
               RewriterBase &rw) {
    RowTile(op, cast<DictionaryAttr>(a), rw).build();
    return llvm::Error::success();
  };
  unit.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
