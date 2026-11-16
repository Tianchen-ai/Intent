#ifndef INTENT_DIALECT_Block_ANALYSIS_INDEXPREDICATES_H
#define INTENT_DIALECT_Block_ANALYSIS_INDEXPREDICATES_H

#include "Intent/Dialect/Block/IR/BlockOps.h"
#include <optional>

namespace intent::block {

// Prove the comparison for every physical lane from current ranges and loops.
// An absent result leaves the original predicate and all its users intact.
std::optional<bool> proveRangeComparison(CompareOp comparison);

// Inclusive upper bound and comparison limit, both specialization expressions.
// This is an implication guard: upper < limit implies the original comparison;
// it does not assert that arbitrary runtime shapes satisfy that relationship.
struct IndexComparisonBound {
  PhysicalExprAttr upper;
  PhysicalExprAttr limit;
};
std::optional<IndexComparisonBound> queryIndexComparisonBound(CompareOp comparison);

} // namespace intent::block

#endif
