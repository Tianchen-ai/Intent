#include "../capabilities/Index.h"
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
          auto caps =
              ctx.target.services.get<intentdsl::cutile::ArrayIndices>();
          return intentdsl::block::matrixIssues(
              o, c, ctx, caps->scaledMma ? 16 : 64, 8, 16);
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
