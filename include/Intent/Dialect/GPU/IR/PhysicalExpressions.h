#ifndef INTENT_DIALECT_GPU_IR_PHYSICALEXPRESSIONS_H
#define INTENT_DIALECT_GPU_IR_PHYSICALEXPRESSIONS_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace intent::gpu {

// Resolve only symbolic leaves. Arithmetic has one checked interpretation for
// IR verification, shared analysis and provider candidate instantiation. An
// absent binding, division by zero or unrepresentable result is not a constant.
std::optional<int64_t> evaluatePhysicalExpression(
    PhysicalExprAttr expression,
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf);
std::optional<int64_t> constantPhysicalExpression(PhysicalExprAttr expression);

} // namespace intent::gpu
#endif
