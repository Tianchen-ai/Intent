#ifndef INTENT_DIALECT_GPU_ANALYSIS_UNIFORMVALUES_H
#define INTENT_DIALECT_GPU_ANALYSIS_UNIFORMVALUES_H

#include "Intent/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"

namespace intent::gpu {
UniformExpression describeUniformValue(mlir::Value value);
mlir::Type uniformElementType(mlir::Type type);
// Return the scalar SSA leaf only when every intervening fragment operation
// forwards a uniform value. A reshape of varying data has no scalar leaf.
mlir::Value uniformScalarSource(mlir::Value value);
// Prove independently which fragment axes do not change the element value.
// Facts follow explicit axis relations through pure value computations. False
// includes unknown; a one-lane physical extent alone is not a uniformity proof.
llvm::SmallVector<bool> uniformFragmentAxes(mlir::Value value);
}
#endif
