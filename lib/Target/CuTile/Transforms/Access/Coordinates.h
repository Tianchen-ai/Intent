#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_COORDINATES_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_COORDINATES_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace intent::cutile {

mlir::Value uniformScalarFill(mlir::Value value);
mlir::FailureOr<llvm::SmallVector<mlir::Value>>
orderedCoordinates(gpu::AccessOpInterface access);
mlir::FailureOr<llvm::SmallVector<mlir::Value>>
materializeCoordinateDomains(mlir::OpBuilder &builder,
                             gpu::AccessOpInterface access);
mlir::FailureOr<unsigned> nativeAccessRangeAxis(gpu::AccessOpInterface access,
                                               unsigned coordinateIndex,
                                               gpu::MakeRangeOp range);
mlir::OpBuilder prepareBranch(mlir::Region &region);
bool isIdentityPermutation(llvm::ArrayRef<int64_t> permutation);
llvm::SmallVector<int64_t> identityAxes(unsigned rank);

} // namespace intent::cutile

#endif
