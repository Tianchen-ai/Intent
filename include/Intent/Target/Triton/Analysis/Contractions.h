#ifndef INTENT_TARGET_TRITON_ANALYSIS_CONTRACTIONS_H
#define INTENT_TARGET_TRITON_ANALYSIS_CONTRACTIONS_H

#include "Intent/Dialect/Block/IR/BlockOps.h"

namespace intent::triton {

// The provider's existing short-reduction form policy, shared by selection and
// current-program resource checks; not a cross-SDK contraction capability.
inline constexpr int64_t expansionReductionThreshold = 16;

block::PhysicalExprAttr expandedContractionElements(block::ContractOp contract);
block::ConfigurationRequirementAttr
contractionExpansionRequirement(block::ContractOp contract);

} // namespace intent::triton

#endif
