#ifndef INTENT_DIALECT_Block_TRANSFORMS_REGION_REALIZATION_H
#define INTENT_DIALECT_Block_TRANSFORMS_REGION_REALIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::block {

mlir::LogicalResult realizeRegionFolds(mlir::ModuleOp module,
                                     bool simplifyFirstSummary);
mlir::LogicalResult realizeRegionScans(mlir::ModuleOp module);

} // namespace intent::block
#endif
