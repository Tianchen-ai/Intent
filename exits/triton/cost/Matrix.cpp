#include "intentdsl/levels/execution/block/ir/MatrixIssue.h"
namespace CI_UNIT_NAMESPACE {
void matrixIssues(compilerinfra::Unit &u) {
  u.costModels.push_back(
      {{"matrix-issues", "levels/execution/block::contraction", "1",
        [](mlir::Operation *o, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &) {
          return intentdsl::block::matrixIssues(o, c, 16, 8, 16);
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
