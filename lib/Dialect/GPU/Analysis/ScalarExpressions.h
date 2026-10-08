#ifndef INTENT_GPU_ANALYSIS_SCALAREXPRESSIONS_H
#define INTENT_GPU_ANALYSIS_SCALAREXPRESSIONS_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

namespace intent::gpu::detail {

std::optional<int64_t> integerConstant(mlir::Value value);
mlir::Value stripBroadcast(mlir::Value value);
mlir::Value stripScalarIdentity(mlir::Value value);
bool sameScalarExpression(mlir::Value lhs, mlir::Value rhs, unsigned depth = 0);
bool isUnitStepValue(mlir::Value value);
PhysicalExprAttr resourceExtentExpression(mlir::Value resource, unsigned axis);
bool valueMatchesExtent(mlir::Value value, PhysicalExprAttr extent);

} // namespace intent::gpu::detail

#endif
