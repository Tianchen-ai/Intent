#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "mlir/IR/PatternMatch.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
using namespace intentdsl;
void registerContraction(Unit &u) {
  DecisionFamily family;
  family.name = "contraction";
  family.applies = [](Operation *op) {
    return op->getName().getStringRef() == "block.matrix_domain" &&
           !op->hasAttr("block.output_tile");
  };
  family.read = [](Operation *op) { return op->getAttr("block.contraction"); };
  family.write = [](Operation *op, Attribute a) {
    op->setAttr("block.contraction", a);
    return llvm::Error::success();
  };
  family.verify = [](Operation *op, Attribute a,
                     const TargetProfile &p) -> llvm::Error {
    if (!block::supportsMatrixTile(op, a, p))
      return llvm::createStringError("unsupported contraction tile");
    return llvm::Error::success();
  };
  family.verifyCandidate = [](Operation *, const StructuralCandidate &c,
                              const TargetProfile &) -> llvm::Error {
    auto tile = cast<DenseI64ArrayAttr>(c.value);
    std::string id = std::to_string(tile[0]) + "x" + std::to_string(tile[1]) +
                     "x" + std::to_string(tile[2]);
    if (c.id != id || c.topology != "output-stationary" ||
        c.preconditions != std::vector<std::string>{"independent-output-tiles"})
      return llvm::createStringError("candidate and matrix tile disagree");
    return llvm::Error::success();
  };
  u.families.push_back({{std::move(family), "block@decided", "block@realized"},
                        COMPILERINFRA_SOURCE});
  Strategy strategy;
  strategy.name = "matrix-tiles";
  strategy.family = u.qualify("contraction");
  strategy.enumerate = [](Operation *op, const TargetProfile &p,
                          AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    auto caps = p.services.get<block::MatrixTarget>();
    if (!caps)
      return llvm::createStringError("exit has no contraction implementation");
    Builder b(op->getContext());
    std::vector<StructuralCandidate> rows;
    for (auto &t : caps->tiles) {
      if (!t.operation.empty() &&
          llvm::none_of(op->getRegion(0).front(), [&](Operation &entry) {
            return entry.getName().stripDialect() == t.operation;
          }))
        continue;
      rows.push_back({std::to_string(t.m) + "x" + std::to_string(t.n) + "x" +
                          std::to_string(t.k),
                      "output-stationary",
                      b.getDenseI64ArrayAttr({t.m, t.n, t.k}),
                      {"independent-output-tiles"}});
    }
    return rows;
  };
  u.strategies.push_back({std::move(strategy), COMPILERINFRA_SOURCE});
  Realizer realize;
  realize.name = "tile-matrix";
  realize.family = u.qualify("contraction");
  realize.apply = [](Operation *op, Attribute a, const TargetProfile &p,
                     RewriterBase &b) -> llvm::Error {
    auto facts = p.services.get<CallFacts>();
    auto shape = block::matrixShape(op, *facts);
    if (!shape)
      return shape.takeError();
    auto fn = op->getParentOfType<func::FuncOp>();
    auto tile = cast<DenseI64ArrayAttr>(a);
    for (auto [i, value] : llvm::enumerate(fn.getArguments())) {
      SmallVector<int64_t> strides;
      for (auto &v : *facts->arguments[i].getAsObject()->getArray("strides"))
        strides.push_back(*v.getAsInteger());
      fn.setArgAttr(i, "block.strides", b.getDenseI64ArrayAttr(strides));
      SmallVector<int64_t> extents;
      for (auto &v : *facts->shapeOf(value))
        extents.push_back(*v.getAsInteger());
      fn.setArgAttr(i, "block.shape", b.getDenseI64ArrayAttr(extents));
    }
    op->setAttr("block.output_tile", tile);
    op->setAttr(
        "block.extents",
        b.getDenseI64ArrayAttr({shape->m, shape->n, shape->k, shape->batches}));
    auto &body = op->getRegion(0).front();
    for (auto &calc : llvm::make_early_inc_range(body)) {
      auto n = calc.getName().stripDialect();
      if (n != "contract" && n != "scaled_contract" && n != "q4k_dot")
        continue;
      b.setInsertionPoint(&calc);
      OperationState loop(calc.getLoc(), "block.panel_fold");
      loop.addAttribute("bound", b.getI64IntegerAttr(shape->k));
      loop.addAttribute("step", b.getI64IntegerAttr(tile[2]));
      loop.addTypes(calc.getResultTypes());
      loop.addRegion();
      auto *fold = b.create(loop);
      calc.getResult(0).replaceAllUsesWith(fold->getResult(0));
      auto &panel = fold->getRegion(0).emplaceBlock();
      calc.moveBefore(&panel, panel.end());
      b.setInsertionPointToEnd(&panel);
      OperationState yield(calc.getLoc(), "block.panel_yield");
      yield.addOperands(calc.getResult(0));
      b.create(yield);
    }
    for (auto &r : p.services.get<block::MatrixTarget>()->tiles)
      if (tile.asArrayRef() == ArrayRef<int64_t>{r.m, r.n, r.k})
        op->setAttr("block.native_row", b.getStringAttr(r.native));
    op->removeAttr("block.contraction");
    return llvm::Error::success();
  };
  u.realizers.push_back({std::move(realize), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
