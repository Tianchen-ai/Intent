#ifndef INTENT_DIALECT_CPU_TRANSFORMS_REGION_REGIONS_H
#define INTENT_DIALECT_CPU_TRANSFORMS_REGION_REGIONS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cpu {

struct Configuration;
class ImplementationRegistry;

mlir::LogicalResult groupRegionComputations(
    mlir::func::FuncOp function, const Configuration &configuration);
mlir::LogicalResult realizeRegions(
    mlir::func::FuncOp function, const Configuration &configuration,
    const ImplementationRegistry &implementations, bool simplifyFirstSummary);

} // namespace intent::cpu
#endif
