#include "compilerinfra/Assembly.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/IRMapping.h"
#include "Dialect.h.inc"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
void dialect(DialectRegistry &r) {
  r.insert<intentdsl::rowmojo::RowMojoDialect>();
}
namespace {
class Columns {
  OpBuilder b;
  Operation *task;
  Location loc;
  IRMapping reduced, tile;
  Value col;
  int64_t width;
  Operation *make(StringRef name, ValueRange args, TypeRange types,
                  ArrayRef<NamedAttribute> attrs = {}) {
    OperationState s(loc, ("mojo_row." + name).str());
    s.addOperands(args);
    s.addTypes(types);
    s.addAttributes(attrs);
    return b.create(s);
  }
  Value expression(Value v) {
    if (auto known = reduced.lookupOrNull(v))
      return known;
    if (auto known = tile.lookupOrNull(v))
      return known;
    auto *op = v.getDefiningOp();
    if (!op)
      return v;
    auto n = op->getName().stripDialect();
    Type t = v.getType();
    if (auto ranked = dyn_cast<RankedTensorType>(t))
      t = VectorType::get({width}, ranked.getElementType());
    Value result;
    if (n == "read")
      result = make("read",
                    {op->getOperand(0),
                     task->getRegion(0).front().getArgument(0), col},
                    t)
                   ->getResult(0);
    else {
      SmallVector<Value> args;
      for (auto value : op->getOperands())
        args.push_back(expression(value));
      result =
          make("apply", args, t, b.getNamedAttr("kind", b.getStringAttr(n)))
              ->getResult(0);
    }
    tile.map(v, result);
    return result;
  }
  Value sweep(Operation *op, StringRef range, Value prev, int64_t lanes) {
    OpBuilder::InsertionGuard restore(b);
    width = lanes;
    tile.clear();
    bool store = op->getName().stripDialect() == "write";
    OperationState s(loc, "mojo_row.sweep");
    if (prev)
      s.addOperands(prev);
    if (!store)
      s.addTypes(b.getF32Type());
    s.addAttribute("width", b.getI64IntegerAttr(width));
    s.addAttribute("range", b.getStringAttr(range));
    s.addAttribute(
        "algebra",
        b.getStringAttr(store ? "store" : op->getName().stripDialect()));
    s.addRegion();
    auto *loop = b.create(s);
    auto &bb = loop->getRegion(0).emplaceBlock();
    col = bb.addArgument(b.getIndexType(), loc);
    b.setInsertionPointToEnd(&bb);
    if (store) {
      make("write",
           {op->getOperand(0), op->getOperand(1), col,
            expression(op->getOperand(2))},
           {});
      make("yield", {}, {});
      return {};
    }
    make("yield", expression(op->getOperand(0)), {});
    return loop->getResult(0);
  }

public:
  Columns(Operation *op) : b(op), task(op), loc(op->getLoc()) {}
  void run(int64_t lanes) {
    auto &bb = task->getRegion(0).front();
    SmallVector<Operation *> original;
    for (auto &op : bb.without_terminator())
      original.push_back(&op);
    b.setInsertionPoint(&bb.back());
    for (auto *op : original) {
      auto n = op->getName().stripDialect();
      if (n != "sum" && n != "max_num" && n != "write")
        continue;
      auto full = sweep(op, "full", {}, lanes);
      auto done = sweep(op, "tail", full, 1);
      if (done)
        reduced.map(op->getResult(0), done);
    }
    for (auto *op : llvm::reverse(original))
      op->erase();
    task->setAttr("mojo_row.schedule", b.getStringAttr("static-row-groups"));
    task->setAttr("mojo_row.lanes", b.getI64IntegerAttr(lanes));
  }
};
struct Realize : PassWrapper<Realize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Realize)
  int64_t lanes;
  explicit Realize(int64_t lanes) : lanes(lanes) {}
  void runOnOperation() final {
    getContext().getOrLoadDialect<intentdsl::rowmojo::RowMojoDialect>();
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "cpu.task")
        Columns(op).run(lanes);
    });
  }
};
} // namespace
void realize(OpPassManager &pm, int64_t lanes) {
  pm.addPass(std::make_unique<Realize>(lanes));
}
} // namespace CI_UNIT_NAMESPACE
