#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
namespace CI_UNIT_NAMESPACE {
bool canNarrowAddress(mlir::Operation &);
mlir::LogicalResult
verifyTileProgram(mlir::func::FuncOp fn,
                  const compilerinfra::TargetProfile &profile) {
  return intentdsl::block::verifyTileProgram(fn, profile, canNarrowAddress);
}
} // namespace CI_UNIT_NAMESPACE
