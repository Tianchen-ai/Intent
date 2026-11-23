#include "intentdsl/levels/execution/block/ir/MatrixIssue.h"
namespace CI_UNIT_NAMESPACE {
void matrixIssues(compilerinfra::Unit &u) {
  u.costModels.push_back(
      {{"collective-steps", "levels/execution/block::ordered-owners", "1",
        [](mlir::Operation *, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &) {
          compilerinfra::CostEstimate estimate;
          estimate.scope = "one collective";
          if (auto d = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(c.value))
            if (auto span = d.getAs<mlir::IntegerAttr>("span");
                span && span.getInt() > 0)
              estimate.terms["reduction_steps"] = {
                  std::ceil(std::log2(double(span.getInt()))), "count"};
          return estimate;
        }},
       COMPILERINFRA_SOURCE});
  u.costModels.push_back(
      {{"matrix-issues", "levels/execution/block::contraction", "1",
        [](mlir::Operation *o, const compilerinfra::StructuralCandidate &c,
           compilerinfra::CostContext &) {
          return intentdsl::block::matrixIssues(o, c, 16, 8, 16);
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
