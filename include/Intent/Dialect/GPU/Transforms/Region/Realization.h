#ifndef INTENT_DIALECT_GPU_TRANSFORMS_REGION_REALIZATION_H
#define INTENT_DIALECT_GPU_TRANSFORMS_REGION_REALIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu {

mlir::LogicalResult realizeRegionFolds(mlir::ModuleOp module,
                                     bool simplifyFirstSummary);
mlir::LogicalResult realizeRegionScans(mlir::ModuleOp module);

} // namespace intent::gpu
#endif
