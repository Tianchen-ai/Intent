#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "mlir/IR/IRMapping.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
class PanelBuilder {
  OpBuilder b;
  Block &task;
  IRMapping summaries, panel;
  Value offset, extent;
  Operation *make(StringRef name, ValueRange operands, TypeRange results) {
    OperationState s(task.getParentOp()->getLoc(), name);
    s.addOperands(operands);
    s.addTypes(results);
    return b.create(s);
  }
  Value local(Value original) {
    if (auto v = summaries.lookupOrNull(original))
      return v;
    if (auto v = panel.lookupOrNull(original))
      return v;
    auto *op = original.getDefiningOp();
    if (!op)
      return original;
    if (op->getName().getStringRef() == "cpu.read") {
      auto result =
          make("weft_host.read_panel",
               {op->getOperand(0), task.getArgument(0), offset, extent},
               original.getType())
              ->getResult(0);
      panel.map(original, result);
      return result;
    }
    for (auto value : op->getOperands())
      panel.map(value, local(value));
    return b.clone(*op, panel)->getResult(0);
  }

public:
  explicit PanelBuilder(Block &task) : b(task.getParentOp()), task(task) {}
  void run() {
    SmallVector<Operation *> old;
    for (auto &op : task.without_terminator())
      old.push_back(&op);
    for (auto *op : old) {
      auto name = op->getName().stripDialect();
      bool store = name == "write";
      if (!store && name != "sum" && name != "max_num")
        continue;
      b.setInsertionPoint(&task.back());
      OperationState s(op->getLoc(), "weft_host.sweep");
      s.addAttribute("combine", b.getStringAttr(store ? "store" : name));
      if (!store)
        s.addTypes(b.getF32Type());
      s.addRegion();
      auto *sweep = b.create(s);
      auto &body = sweep->getRegion(0).emplaceBlock();
      offset = body.addArgument(b.getIndexType(), op->getLoc());
      extent = body.addArgument(b.getIndexType(), op->getLoc());
      b.setInsertionPointToEnd(&body);
      panel.clear();
      if (store) {
        make("weft_host.write_panel",
             {op->getOperand(0), task.getArgument(0), offset, extent,
              local(op->getOperand(2))},
             {});
        make("weft_host.yield", {}, {});
      } else {
        auto partial = local(op->getResult(0));
        make("weft_host.yield", partial, {});
        summaries.map(op->getResult(0), sweep->getResult(0));
      }
    }
    for (auto *op : llvm::reverse(old))
      op->erase();
  }
};
} // namespace
void buildPanels(Operation *task) {
  PanelBuilder(task->getRegion(0).front()).run();
}
LogicalResult verifyPanels(Operation *dispatch) {
  auto &body = dispatch->getRegion(0).front();
  for (auto &s : body.without_terminator()) {
    if (s.getName().getStringRef() != "weft_host.sweep")
      return s.emitError("expected a column panel sweep");
    auto kind = s.getAttrOfType<StringAttr>("combine");
    bool store = kind == "store";
    if ((!store && kind != "max_num" && kind != "sum") ||
        s.getNumResults() != unsigned(!store))
      return s.emitError("invalid sweep combine");
    auto &bb = s.getRegion(0).front();
    if (bb.getNumArguments() != 2 || !bb.getArgument(0).getType().isIndex() ||
        !bb.getArgument(1).getType().isIndex() || bb.empty() ||
        bb.back().getName().getStringRef() != "weft_host.yield" ||
        bb.back().getNumOperands() != unsigned(!store))
      return s.emitError("invalid panel indices or reduction yield");
    unsigned writes = 0;
    for (auto &op : bb.without_terminator()) {
      auto n = op.getName().stripDialect();
      if (n == "read_panel" || n == "write_panel") {
        bool write = n == "write_panel";
        writes += write;
        auto view = cast<MemRefType>(op.getOperand(0).getType());
        if (op.getOperand(1) != body.getArgument(0) ||
            op.getOperand(2) != bb.getArgument(0) ||
            op.getOperand(3) != bb.getArgument(1) ||
            (write ? op.getOperand(4).getType() : op.getResult(0).getType()) !=
                RankedTensorType::get({ShapedType::kDynamic},
                                      view.getElementType()))
          return op.emitError("panel access does not cover the active members");
      } else if (op.getName().getDialectNamespace() != "cpu" || n == "read" ||
                 n == "write" || n == "parallel" || n == "task" || n == "join")
        return op.emitError("unexpected instruction inside Weft panel");
    }
    if (writes != unsigned(store))
      return s.emitError("wrong panel write count");
    if (!store) {
      auto *local = bb.back().getOperand(0).getDefiningOp();
      if (!local || local->getBlock() != &bb ||
          local->getName().stripDialect() != kind.getValue())
        return s.emitError("panel and cross-panel combine differ");
    }
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
