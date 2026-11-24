#include "intentdsl/levels/execution/block/ir/FoldIssue.h"
#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
#include "intentdsl/levels/execution/block/ir/SharedReads.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
using namespace mlir;
using namespace compilerinfra;
namespace CI_UNIT_NAMESPACE {
void registerMixedOwners(Unit &u) {
  Strategy s;
  s.name = "mixed-owner-reads";
  s.family = u.qualify("ordered-owners");
  s.enumerate = [](Operation *o, const TargetProfile &p, AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    std::vector<StructuralCandidate> candidates;
    Builder b(o->getContext());
    for (int rows : {4, 8, 16, 32, 64, 128})
      for (int span : {32, 64, 128})
        if (auto axis = intentdsl::block::issueAxis(o, p, rows, span)) {
          auto d = b.getDictionaryAttr(
              {b.getNamedAttr("memory", b.getStringAttr("matrix-fold")),
               b.getNamedAttr("width", b.getI64IntegerAttr(rows)),
               b.getNamedAttr("span", b.getI64IntegerAttr(span)),
               b.getNamedAttr("axis", b.getI64IntegerAttr(*axis))});
          candidates.push_back(
              {"pair-" + std::to_string(rows) + "x" + std::to_string(span),
               "fold-matrix-issue",
               d,
               {"disjoint-owners"}});
        }
    for (int span : {16, 32, 64, 128, 256, 512, 1024, 2048, 4096})
      if (intentdsl::block::panelFolds(o, p, span)) {
        auto d = b.getDictionaryAttr(
            {b.getNamedAttr("memory", b.getStringAttr("token")),
             b.getNamedAttr("width", b.getI64IntegerAttr(1)),
             b.getNamedAttr("span", b.getI64IntegerAttr(span))});
        candidates.push_back({"tokens-" + std::to_string(span),
                              "parallel-fold-contributions",
                              d,
                              {"disjoint-owners"}});
      }
    auto facts = p.services.get<intentdsl::CallFacts>();
    auto reads = intentdsl::block::sharedReads(o, *facts);
    if (!reads)
      return candidates;
    // A dot reduction and a carried member accumulator consume the same
    // projected input region. Both must use one owner map.
    bool dot = false, vectorCarry = false;
    o->walk([&](Operation *inst) {
      if (inst->getName().stripDialect() == "lane_sum")
        dot |= bool(inst->getOperand(0).getDefiningOp<arith::MulFOp>());
      if (auto loop = dyn_cast<scf::ForOp>(inst))
        vectorCarry |= llvm::any_of(loop.getInitArgs(), [](Value v) {
          return isa<RankedTensorType>(v.getType());
        });
    });
    if (!dot || !vectorCarry)
      return candidates;
    for (int w = 2; w <= 32; w *= 2) {
      auto d = b.getDictionaryAttr(
          {b.getNamedAttr("memory", b.getStringAttr("shared")),
           b.getNamedAttr("axis", b.getI64IntegerAttr(reads->axis)),
           b.getNamedAttr("width", b.getI64IntegerAttr(w))});
      if (intentdsl::block::legalSharedReads(o, p, d))
        candidates.push_back({"shared-" + std::to_string(w),
                              "shared-owner-reads",
                              d,
                              {"disjoint-owners"}});
    }
    return candidates;
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
