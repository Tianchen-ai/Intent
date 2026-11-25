#include "intentdsl/levels/execution/block/ir/CollectiveIssue.h"
#include "intentdsl/levels/execution/block/ir/MatrixIssue.h"
namespace CI_UNIT_NAMESPACE {
void matrixIssues(compilerinfra::Unit &u) {
  u.costModels.push_back(
      {intentdsl::block::collectiveSteps(), COMPILERINFRA_SOURCE});
  u.costModels.push_back(
      {{"matrix-issues", "levels/execution/block::contraction", "1",
        [](mlir::Operation *o, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &ctx) {
          return intentdsl::block::matrixIssues(o, c, ctx, 16, 8, 16, 2);
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
