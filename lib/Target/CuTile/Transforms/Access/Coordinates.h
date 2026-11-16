#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_COORDINATES_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_COORDINATES_H

#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace intent::cutile {

mlir::Value uniformScalarFill(mlir::Value value);
mlir::FailureOr<llvm::SmallVector<mlir::Value>>
orderedCoordinates(block::AccessOpInterface access);
mlir::FailureOr<llvm::SmallVector<mlir::Value>>
materializeCoordinateDomains(mlir::OpBuilder &builder,
                             block::AccessOpInterface access);
mlir::FailureOr<unsigned> nativeAccessRangeAxis(block::AccessOpInterface access,
                                               unsigned coordinateIndex,
                                               block::MakeRangeOp range);
mlir::OpBuilder prepareBranch(mlir::Region &region);
bool isIdentityPermutation(llvm::ArrayRef<int64_t> permutation);
llvm::SmallVector<int64_t> identityAxes(unsigned rank);

} // namespace intent::cutile

#endif
