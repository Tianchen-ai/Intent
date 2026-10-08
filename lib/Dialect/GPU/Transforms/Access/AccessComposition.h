#ifndef INTENT_GPU_TRANSFORMS_ACCESSCOMPOSITION_H
#define INTENT_GPU_TRANSFORMS_ACCESSCOMPOSITION_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/IR/IRMapping.h"

namespace intent::gpu::access {

bool isZero(mlir::Value value);

// Keep existing coordinate/validity/fill values over the untouched source
// axes. These are numeric SSA snapshots, not permission to repeat their reads.
void bindUnchangedAccessValues(
    mlir::OpBuilder &builder, mlir::ValueRange values, FragmentType source,
    FragmentType target, llvm::ArrayRef<int64_t> slicedAxes,
    mlir::IRMapping &mapping);

mlir::FailureOr<mlir::Value> replayFragmentValue(
    mlir::OpBuilder &builder, mlir::Value value, FragmentType target,
    mlir::IRMapping &mapping, PhysicalProgramAnalysis &analysis,
    mlir::Operation *insertionAnchor);

mlir::FailureOr<mlir::Value> replayScalarValue(
    mlir::OpBuilder &builder, mlir::Value value, mlir::IRMapping &mapping,
    PhysicalProgramAnalysis &analysis, mlir::Operation *insertionAnchor);

mlir::FailureOr<mlir::Value> combinePredicates(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Type valueType,
    mlir::Value lhs, mlir::Value rhs);

mlir::FailureOr<mlir::Value> gatherBounds(
    mlir::OpBuilder &builder, mlir::Location location, FragmentType source,
    mlir::ValueRange coordinates, llvm::ArrayRef<int64_t> axes,
    mlir::Type indexType);

bool foldIndexRecompositions(mlir::func::FuncOp kernel);

bool sameUniformValue(mlir::Value lhs, mlir::Value rhs);

bool unitAxes(FragmentType type, llvm::ArrayRef<unsigned> axes);

mlir::FailureOr<bool> composeSelectLoad(SelectOp select);

mlir::FailureOr<bool> composeReshapedGather(GatherOp gather);

mlir::FailureOr<bool> composeRangeGather(GatherOp gather);

mlir::FailureOr<bool> composePointwiseGather(GatherOp gather);

mlir::FailureOr<bool> composeReducedGather(GatherOp gather);

mlir::Value cancelIndexOffset(mlir::OpBuilder &builder, mlir::Value coordinate,
                             mlir::Value offset);

mlir::FailureOr<bool> composeLoadGather(GatherOp gather);

bool reuseFragmentGather(GatherOp gather);

mlir::FailureOr<bool> composeIdentityFragmentGather(GatherOp gather);

mlir::FailureOr<bool> composeReductionGathers(ReduceOp reduce);

mlir::FailureOr<bool> projectFragmentGather(GatherOp gather);

mlir::FailureOr<bool> composeBroadcastGather(GatherOp gather);

bool composeReshapedPointwise(ReshapeOp reshape);

mlir::FailureOr<bool> composeReshapedLoad(ReshapeOp reshape);

mlir::FailureOr<bool> composeReshapedStore(StoreOp store);

bool reuseStableLoads(mlir::func::FuncOp kernel);

void sinkStableLoadChains(mlir::func::FuncOp kernel);

} // namespace intent::gpu::access

#endif
