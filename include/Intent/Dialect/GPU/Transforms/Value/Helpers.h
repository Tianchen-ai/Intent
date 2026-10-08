#ifndef INTENT_DIALECT_GPU_TRANSFORMS_VALUE_HELPERS_H
#define INTENT_DIALECT_GPU_TRANSFORMS_VALUE_HELPERS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LogicalResult.h"
#include <string>

namespace intent::gpu {

// Rebuild a proved lane-wise helper with explicitly selected formal/yield types.
// Constants stay scalar until a use needs a fragment. The temporary region is
// published only after every operation and yield has been reconstructed.
mlir::LogicalResult cloneLaneWiseHelper(mlir::Region &source, mlir::Region &target,
                                       mlir::TypeRange argumentTypes,
                                       mlir::TypeRange resultTypes,
                                       std::string &reason);
mlir::LogicalResult scalarizeElementwiseCallback(mlir::Region &source,
                                                mlir::Region &target);
mlir::LogicalResult liftCombineRegion(mlir::Region &source, mlir::Region &target,
                                     mlir::TypeRange accumulatorTypes,
                                     std::string &reason);
// Retype a single-component, capture-free reduction's retained lane domain.
// Source and identity projection plus helper reconstruction are transactional;
// reduction members/order/attributes remain those of the original operation.
mlir::FailureOr<mlir::Value> projectLaneWiseReduction(
    mlir::OpBuilder &builder, mlir::Location location, ReduceOp reduction,
    FragmentType target);

} // namespace intent::gpu
#endif
