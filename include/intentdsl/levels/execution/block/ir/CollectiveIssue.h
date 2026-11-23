#pragma once
#include "intentdsl/levels/execution/block/ir/MatrixIssue.h"
namespace intentdsl::block {
inline compilerinfra::CostModel collectiveSteps() {
  return {"collective-steps", "levels/execution/block::ordered-owners", "1",
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
          }};
}
} // namespace intentdsl::block
