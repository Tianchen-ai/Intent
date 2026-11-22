#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::dsa {
namespace {
// The exit supplies one physical implementation of the two input panels and
// its accumulator. All sizes include the consumer's padded native footprint.
int64_t bytes(const block::MatrixAccess &a, const TargetProfile &profile) {
  auto t = a.tile;
  return profile.services.get<MatrixStorage>()->privateBytes(
      t[0], t[1], t[2], a.dot.getName().stripDialect());
}
SmallVector<std::pair<Value, std::string>> feeds(block::MatrixAccess &a) {
  auto &dot = a.dot;
  auto n = dot.getName().stripDialect();
  if (n == "window_contract")
    return {{dot.getOperand(0), "window-left"},
            {dot.getOperand(1), "window-right"},
            {dot.getOperand(2), "row-coefficient"},
            {dot.getOperand(3), "row-prefix"}};
  if (n == "q4k_dot")
    return {{dot.getOperand(0), "packed-weight"},
            {dot.getOperand(1).getDefiningOp()->getOperand(0),
             "encode-activation"}};
  if (n == "scaled_contract")
    return {{dot.getOperand(0), "left-groups"},
            {dot.getOperand(1), "left-scales"},
            {dot.getOperand(2), "right-groups"},
            {kir::RightScale(dot).view, "right-scales"}};
  return {{dot.getOperand(0), "left-panel"},
          {dot.getOperand(1), "right-panel"}};
}
bool supportsSupply(block::MatrixAccess &a, Attribute v,
                    const TargetProfile &p) {
  auto count = dyn_cast_or_null<IntegerAttr>(v);
  auto cap = p.services.get<LocalSupply>();
  return count && cap && p.services.get<MatrixStorage>() && count.getInt() > 0 &&
         count.getInt() <= cap->consumers &&
         bytes(a,p) <= std::min(cap->capacity, cap->budget);
}
} // namespace
LogicalResult verifyMatrix(func::FuncOp fn, const TargetProfile &p,
                           bool supplied) {
  if (!supplied) {
    if (failed(block::verifyTiledMatrix(fn, p, "block")))
      return failure();
    block::MatrixAccess a(fn);
    if (!supportsSupply(a, fn->getAttr("dsa.matrix_choice"), p))
      return fn.emitError(
          "matrix choice exceeds worker count or private storage");
    return success();
  }
  auto verify = [&](Operation &setup) -> LogicalResult {
    if (setup.getName().getStringRef() != "dsa.panel_supply" ||
        !llvm::hasSingleElement(setup.getRegion(0)))
      return failure();
    block::MatrixAccess a(fn);
    auto cap = p.services.get<LocalSupply>();
    auto tasks = setup.getAttrOfType<IntegerAttr>("participants").getInt();
    auto scratch = setup.getAttrOfType<IntegerAttr>("scratch_bytes").getInt();
    if (!cap || !p.services.get<MatrixStorage>() || tasks < 1 || tasks > cap->consumers || scratch != bytes(a,p) ||
        scratch > std::min(cap->budget, cap->capacity))
      return setup.emitError(
          "matrix inputs or accumulator exceed the physical supply");
    auto sources = feeds(a);
    auto &body = setup.getRegion(0).front();
    if (body.getOperations().size() != sources.size() + 1 ||
        body.back().getName().getStringRef() != "dsa.panel_ready" ||
        body.back().getNumOperands() != sources.size())
      return setup.emitError("panel completion arity");
    for (auto [i, entry] : llvm::enumerate(body.without_terminator())) {
      if (entry.getName().getStringRef() != "dsa.stage_operand" ||
          entry.getOperand(0) != sources[i].first ||
          entry.getAttrOfType<StringAttr>("access") != sources[i].second ||
          entry.getAttrOfType<IntegerAttr>("slot").getInt() != int64_t(i) ||
          body.back().getOperand(i) != entry.getResult(0))
        return entry.emitError(
            "incorrect matrix supply or completion dependency");
    }
    return success();
  };
  return block::verifyTiledMatrix(fn, p, "block", verify);
}
} // namespace intentdsl::dsa
namespace CI_UNIT_NAMESPACE {
using namespace intentdsl;
void matrixSupply(Unit &u) {
  DecisionFamily f;
  f.name = "panel-supply";
  f.applies = [](Operation *op) {
    return kir::matrixDomain(dyn_cast<func::FuncOp>(op));
  };
  f.read = [](Operation *op) { return op->getAttr("dsa.matrix_choice"); };
  f.write = [](Operation *op, Attribute v) {
    op->setAttr("dsa.matrix_choice", v);
    return llvm::Error::success();
  };
  f.verify = [](Operation *op, Attribute v,
                const TargetProfile &p) -> llvm::Error {
    block::MatrixAccess a(cast<func::FuncOp>(op));
    if (!dsa::supportsSupply(a, v, p))
      return llvm::createStringError(
          "illegal participant or private panel footprint");
    return llvm::Error::success();
  };
  f.verifyCandidate = [](Operation *, const StructuralCandidate &c,
                         const TargetProfile &) -> llvm::Error {
    if (c.id !=
            "private-" + std::to_string(cast<IntegerAttr>(c.value).getInt()) ||
        c.topology != "private-panels" ||
        c.preconditions != std::vector<std::string>{"complete-private-tiles"})
      return llvm::createStringError("invalid matrix supply candidate");
    return llvm::Error::success();
  };
  u.families.push_back(
      {{std::move(f), "dsa@formed", "dsa@supplied"}, COMPILERINFRA_SOURCE});
  Strategy s;
  s.name = "task-panels";
  s.family = u.qualify("panel-supply");
  s.enumerate = [](Operation *op, const TargetProfile &p, AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    auto cap = p.services.get<dsa::LocalSupply>();
    std::vector<StructuralCandidate> cs;
    for (auto n = cap->consumers; n > 0; n /= 2)
      cs.push_back({"private-" + std::to_string(n),
                    "private-panels",
                    Builder(op->getContext()).getI64IntegerAttr(n),
                    {"complete-private-tiles"}});
    return cs;
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
  Realizer r;
  r.name = "place-panels";
  r.family = u.qualify("panel-supply");
  r.apply = [](Operation *op, Attribute choice, const TargetProfile &p,
               RewriterBase &b) -> llvm::Error {
    op->getContext()->getOrLoadDialect("dsa");
    block::MatrixAccess a(cast<func::FuncOp>(op));
    b.setInsertionPoint(&a.dot);
    OperationState setup(op->getLoc(), "dsa.panel_supply");
    setup.addAttribute("participants", choice);
    setup.addAttribute("scratch_bytes", b.getI64IntegerAttr(dsa::bytes(a,p)));
    setup.addRegion();
    auto *panel = b.create(setup);
    b.createBlock(&panel->getRegion(0));
    SmallVector<Value> completions;
    for (auto [v, mode] : dsa::feeds(a)) {
      OperationState stage(op->getLoc(), "dsa.stage_operand");
      stage.addOperands(v);
      stage.addTypes(b.getIndexType());
      stage.addAttribute("slot", b.getI64IntegerAttr(completions.size()));
      stage.addAttribute("access", b.getStringAttr(mode));
      completions.push_back(b.create(stage)->getResult(0));
    }
    OperationState ready(op->getLoc(), "dsa.panel_ready");
    ready.addOperands(completions);
    b.create(ready);
    op->removeAttr("dsa.matrix_choice");
    return llvm::Error::success();
  };
  u.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
