#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringSet.h"
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
    if (tile.size() == 4 && tile[3] == 2) {
      if (c.id != "split-library" || c.topology != "split-dequant-library" ||
          c.preconditions !=
              std::vector<std::string>{"independent-output-tiles",
                                       "split-grouped-rhs-workspace"})
        return llvm::createStringError("invalid split library candidate");
      return llvm::Error::success();
    }
    if (tile.size() == 4 && tile[3] == 1) {
      std::string id = "split-" + std::to_string(tile[0]) + "x" +
                       std::to_string(tile[1]) + "x" + std::to_string(tile[2]);
      if (c.id != id || c.topology != "split-dequant-contraction" ||
          c.preconditions !=
              std::vector<std::string>{"independent-output-tiles",
                                       "split-grouped-rhs-workspace"})
        return llvm::createStringError("invalid split contraction candidate");
      return llvm::Error::success();
    }
    std::string id = std::to_string(tile[0]) + "x" + std::to_string(tile[1]) +
                     "x" + std::to_string(tile[2]);
    if (tile.size() != 3 || c.id != id || c.topology != "output-stationary" ||
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
    std::vector<std::string> implementations;
    for (auto &operation : op->getRegion(0).front()) {
      auto opName = operation.getName().stripDialect();
      auto found = caps->implementations.find(opName.str());
      if (found != caps->implementations.end()) {
        llvm::append_range(implementations, found->second);
      } else if (opName == "quant_contract") {
        if (const auto *spec = kir::quantFormatSpec(operation)) {
          auto addFallback = [&](llvm::StringRef key) {
            auto it = caps->implementations.find(key.str());
            if (it != caps->implementations.end())
              for (const auto &impl : it->second)
                if (!llvm::is_contained(implementations, impl))
                  implementations.push_back(impl);
          };
          if (spec->roleLayout == kir::QuantRoleLayout::RecordGemv) {
            addFallback("q4k_dot");
          } else if (spec->roleLayout == kir::QuantRoleLayout::BlockScaledPair) {
            addFallback("scaled_contract");
            addFallback("contract");
            addFallback("q4k_dot");
          } else {
            addFallback("contract");
          }
        }
      }
    }
    llvm::StringSet<> seen;
    for (auto &t : caps->tiles) {
      auto choice = b.getDenseI64ArrayAttr({t.m, t.n, t.k});
      if (!block::supportsMatrixTile(op, choice, p))
        continue;
      std::string id = std::to_string(t.m) + "x" + std::to_string(t.n) + "x" +
                       std::to_string(t.k);
      if (!seen.insert(id).second)
        continue;
      rows.push_back({std::move(id),
                      "output-stationary",
                      choice,
                      {"independent-output-tiles"},
                      implementations});
    }
    if (block::canSplitGroupedRhs(op, p)) {
      std::vector<std::string> contractImpls;
      if (auto it = caps->implementations.find("contract");
          it != caps->implementations.end())
        contractImpls = it->second;
      llvm::StringSet<> seenSplit;
      for (auto &t : caps->tiles) {
        auto choice = b.getDenseI64ArrayAttr({t.m, t.n, t.k, 1});
        if (!block::supportsMatrixTile(op, choice, p))
          continue;
        std::string id = "split-" + std::to_string(t.m) + "x" +
                         std::to_string(t.n) + "x" + std::to_string(t.k);
        if (!seenSplit.insert(id).second)
          continue;
        rows.push_back({std::move(id),
                        "split-dequant-contraction",
                        choice,
                        {"independent-output-tiles",
                         "split-grouped-rhs-workspace"},
                        contractImpls});
      }
      if (!caps->libraryGemmImpl.empty()) {
        auto choice = b.getDenseI64ArrayAttr({1, 1, 1, 2});
        if (block::supportsMatrixTile(op, choice, p)) {
          rows.push_back({"split-library",
                          "split-dequant-library",
                          choice,
                          {"independent-output-tiles",
                           "split-grouped-rhs-workspace"},
                          {caps->libraryGemmImpl}});
        }
      }
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
    bool isSplit = tile.size() == 4 && (tile[3] == 1 || tile[3] == 2);
    bool isSplitLib = tile.size() == 4 && tile[3] == 2;
    auto outTile = b.getDenseI64ArrayAttr({tile[0], tile[1], tile[2]});
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
    op->setAttr("block.output_tile", outTile);
    if (isSplit)
      op->setAttr("block.split_dequant", b.getUnitAttr());
    if (isSplitLib)
      op->setAttr("block.split_library", b.getUnitAttr());
    op->setAttr(
        "block.extents",
        b.getDenseI64ArrayAttr({shape->m, shape->n, shape->k, shape->batches}));
    auto &body = op->getRegion(0).front();
    for (auto &calc : llvm::make_early_inc_range(body)) {
      auto n = calc.getName().stripDialect();
      if (!kir::isContraction(n))
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
    if (const auto *matched = block::findMatrixTile(op, tile.asArrayRef(), p))
      op->setAttr("block.native_row", b.getStringAttr(matched->native));
    op->removeAttr("block.contraction");
    return llvm::Error::success();
  };
  u.realizers.push_back({std::move(realize), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
