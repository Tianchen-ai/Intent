#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
namespace CI_UNIT_NAMESPACE {
bool canNarrowAddress(mlir::Operation &);
mlir::LogicalResult
verifyTritonGeometry(mlir::func::FuncOp, const compilerinfra::TargetProfile &);
mlir::LogicalResult
verifyTileProgram(mlir::func::FuncOp fn,
                  const compilerinfra::TargetProfile &profile) {
  if (mlir::failed(verifyTritonGeometry(fn, profile)))
    return mlir::failure();
  return intentdsl::block::verifyTileProgram(fn, profile, canNarrowAddress);
}
} // namespace CI_UNIT_NAMESPACE
