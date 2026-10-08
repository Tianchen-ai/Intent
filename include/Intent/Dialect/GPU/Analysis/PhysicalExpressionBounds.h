#ifndef INTENT_DIALECT_GPU_ANALYSIS_PHYSICALEXPRESSIONBOUNDS_H
#define INTENT_DIALECT_GPU_ANALYSIS_PHYSICALEXPRESSIONBOUNDS_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include <optional>

namespace intent::gpu {

// Prove that the complete host expression can be evaluated with checked signed
// 64-bit arithmetic for the current parameter domain. This is distinct from
// the modular value range of an executable integer operation. Unknown includes
// a potentially overflowing intermediate, even if the final result is bounded.
std::optional<mlir::ConstantIntRanges>
queryPhysicalExpressionRange(PhysicalExprAttr expression,
                             mlir::func::FuncOp kernel);

// Signed ordering over current typed host expressions. Mathematical rewrites
// require checked operand evaluation; this does not create runtime guards or
// turn execution-local coordinate facts into host-visible assumptions.
bool physicalExpressionAtMost(PhysicalExprAttr lhs, PhysicalExprAttr rhs,
                              mlir::func::FuncOp kernel);
bool physicalExpressionLessThan(PhysicalExprAttr lhs, PhysicalExprAttr rhs,
                                mlir::func::FuncOp kernel);

} // namespace intent::gpu
#endif
