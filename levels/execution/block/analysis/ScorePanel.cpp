#include "intentdsl/levels/execution/block/ir/ScorePanel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
using namespace mlir;
namespace intentdsl::block {
bool legalScoreTile(const ScorePanel &s, int64_t rows) {
  auto tile = [](int64_t n) {
    return n >= 16 && n <= 256 && llvm::isPowerOf2_64(n);
  };
  return (rows == 16 || rows == 32 || rows == 64) && s.queries > 0 &&
         s.queries % rows == 0 && tile(s.keys) && tile(s.depth) &&
         tile(s.features);
}
std::optional<ScorePanel> scorePanel(Operation *o,
                                     const compilerinfra::TargetProfile &p) {
  auto f = p.services.get<CallFacts>();
  auto machine = p.services.get<OwnerMachine>();
  if (!f || !machine ||
      (machine->parallelMembers < 16 && !p.services.get<ScoreResources>()))
    return {};
  auto s = kir::normalizedContractions(o, *f);
  if (!s)
    return {};
  return s;
}
bool fitsScoreTile(const ScorePanel &s, int64_t rows,
                   const compilerinfra::TargetProfile &p) {
  if (!legalScoreTile(s, rows))
    return false;
  auto caps = p.services.get<ScoreResources>();
  return !caps || caps->supports(s, rows, *p.services.get<CallFacts>());
}
void realizeScoreTile(Operation *o, const ScorePanel &s, int64_t rows) {
  auto &bb = o->getRegion(0).front();
  auto *end = &bb.back();
  while (auto *prev = end->getPrevNode())
    prev->erase();
  OpBuilder b(end);
  auto loc = o->getLoc();
  auto [batch, head, row] =
      std::make_tuple(bb.getArgument(0), bb.getArgument(1), bb.getArgument(2));
  auto mat = [&](int64_t n, Type t) {
    return RankedTensorType::get({rows, n}, t);
  };
  auto inst = [&](StringRef n, ValueRange args, TypeRange result,
                  bool grouped = false) {
    OperationState state(loc, "block." + n.str());
    state.addOperands(args);
    state.addTypes(result);
    if (grouped)
      state.addAttribute("group", b.getI64IntegerAttr(s.group));
    return b.create(state);
  };
  auto scores = inst("score_contract", {s.q, s.k, s.scale, batch, head, row},
                     mat(s.keys, b.getF32Type()), true)
                    ->getResult(0);
  auto weights = inst("causal_exp", {scores, row},
                      {mat(s.keys, b.getBF16Type()),
                       RankedTensorType::get({rows}, b.getF32Type())});
  auto values =
      inst("value_contract", {weights->getResult(0), s.v, batch, head},
           mat(s.features, b.getF32Type()), true)
          ->getResult(0);
  inst("normalize_store",
       {values, weights->getResult(1), s.y, batch, head, row}, {});
  o->setAttr("block.score_panel", b.getUnitAttr());
}
LogicalResult verifyScoreTile(func::FuncOp fn, StringRef dialect,
                              const compilerinfra::TargetProfile &p) {
  auto &space = fn.front().front();
  auto &bb = space.getRegion(0).front();
  if (bb.getOperations().size() != 5 || bb.getNumArguments() != 3)
    return space.emitError("invalid score panel region");
  auto *a = &bb.front(), *w = a->getNextNode(), *v = w->getNextNode(),
       *y = v->getNextNode();
  auto named = [&](Operation *op, StringRef k) {
    return op->getName().getDialectNamespace() == dialect &&
           op->getName().stripDialect() == k;
  };
  if (!named(a, "score_contract") || !named(w, "causal_exp") ||
      !named(v, "value_contract") || !named(y, "normalize_store") ||
      !named(&bb.back(), "end"))
    return space.emitError(
        "missing score, weights, value contraction or output");
  auto width = space.getAttrOfType<IntegerAttr>("block.owner_tile");
  auto group = a->getAttrOfType<IntegerAttr>("group");
  auto facts = p.services.get<CallFacts>();
  auto b = bb.getArgument(0), h = bb.getArgument(1), row = bb.getArgument(2);
  ScorePanel s{a->getOperand(0),
               a->getOperand(1),
               v->getOperand(1),
               y->getOperand(2),
               a->getOperand(2),
               group ? group.getInt() : 0,
               0,
               0,
               0,
               0};
  if (!width || !facts || fn.getNumArguments() != 5 ||
      ValueRange({s.q, s.k, s.v, s.y, s.scale}) !=
          ValueRange(fn.getArguments()) ||
      !s.scale.getType().isF32() || !kir::scoreShape(s, *facts) ||
      !legalScoreTile(s, width.getInt()) ||
      space.getOperands() != ValueRange({s.q, s.q, s.q}) ||
      space.getAttrOfType<DenseI64ArrayAttr>("axes").asArrayRef() !=
          ArrayRef<int64_t>({0, 1, 2}) ||
      v->getAttr("group") != group ||
      a->getOperands().drop_front(3) != ValueRange({b, h, row}) ||
      w->getOperands() != ValueRange({a->getResult(0), row}) ||
      v->getOperands() != ValueRange({w->getResult(0), s.v, b, h}) ||
      y->getOperands() !=
          ValueRange({v->getResult(0), w->getResult(1), s.y, b, h, row}))
    return space.emitError(
        "invalid mixed panel domains or producer-consumer relations");
  if (auto cap = p.services.get<ScoreResources>();
      cap && !cap->supports(s, width.getInt(), *facts))
    return space.emitError(
        "selected mixed panel exceeds storage or stride capabilities");
  auto type = [&](Type t, ArrayRef<int64_t> dims, Type elem) {
    return t == RankedTensorType::get(dims, elem);
  };
  Builder builder(fn.getContext());
  auto m = width.getInt();
  return success(
      type(a->getResult(0).getType(), {m, s.keys}, builder.getF32Type()) &&
      type(w->getResult(0).getType(), {m, s.keys}, builder.getBF16Type()) &&
      type(w->getResult(1).getType(), {m}, builder.getF32Type()) &&
      type(v->getResult(0).getType(), {m, s.features}, builder.getF32Type()));
}
} // namespace intentdsl::block
