#ifndef INTENT_DIALECT_GPU_ANALYSIS_INTEGERRANGES_H
#define INTENT_DIALECT_GPU_ANALYSIS_INTEGERRANGES_H

#include "Intent/Analysis/IntegerRanges.h"

namespace intent::gpu {

// Current GPU parameter, coordinate and resource facts seed the shared
// fixed-width analysis. No mutation or physical parameter choice is made here.
IntegerRangePolicy integerRangePolicy();
std::optional<mlir::ConstantIntRanges> queryIntegerRange(mlir::Value value);
bool isValuePreservingIntegerCast(mlir::Value source, mlir::Type targetType);
bool integerOperationDoesNotWrap(mlir::Value value);

} // namespace intent::gpu
#endif
