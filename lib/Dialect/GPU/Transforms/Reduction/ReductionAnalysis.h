#ifndef INTENT_GPU_TRANSFORMS_REDUCTIONANALYSIS_H
#define INTENT_GPU_TRANSFORMS_REDUCTIONANALYSIS_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace intent::gpu::reduction {

// Read-only reduction queries over the current physical program. SourcePlan
// borrows SSA values only for the duration of a rewrite; it is not execution IR.

struct SourcePlan {
  mlir::Value source;
  PhysicalSourceAxis sourceIdentity;
  unsigned reductionAxis;
  MakeRangeOp reductionRange;
  llvm::SmallVector<LoadOp> roots;
  llvm::SmallVector<MakeRangeOp> ranges;
};

struct RootAccess {
  LoadOp load;
  MakeRangeOp range;
  unsigned coordinateIndex;
  unsigned fragmentAxis;
};

bool isCompileTimeExtent(PhysicalExprAttr expression);

bool isCompileTimeValue(mlir::Value value);

bool exceedsRegisterFile(mlir::Value source, mlir::func::FuncOp kernel);

// Optional bounded supply for a large tuple whose current full producers can
// actually be removed. This nominal footprint is not a native resource limit.
bool shouldTileReductionSources(ReduceOp reduce, mlir::func::FuncOp kernel);

// Proves removable, replayable input supply for bounded outer-axis partials.
// This is a legality query for the static full-ordinal decomposition choice;
// it does not override the default of a deferred full-coverage dimension.
bool canChunkReductionOuter(ReduceOp reduce, mlir::func::FuncOp kernel,
                            unsigned outerAxis);

bool requiresPhysicalRealization(ReduceOp reduce);

mlir::ArrayAttr reductionSources(ReduceOp reduce);

MakeRangeOp sourceRange(mlir::Value value);

mlir::FailureOr<mlir::Value> scalarSource(mlir::Value value);

bool sameScalarValue(mlir::Value lhs, mlir::Value rhs);

// Match the complete typed callback at an actual update. Non-null right members
// are constraints; null members are bound to the corresponding current SSA.
bool matchReductionCombine(ReduceOp reference, mlir::Block *body,
    mlir::ValueRange left, llvm::SmallVectorImpl<mlir::Value> &right,
    mlir::ValueRange results,
    llvm::SmallPtrSetImpl<mlir::Operation *> &matched);

bool isReplayableWithoutLoad(mlir::Value value, PhysicalSourceAxis source);

mlir::FailureOr<std::optional<RootAccess>>
analyzeRoot(LoadOp load, llvm::ArrayRef<MakeRangeOp> reductionRanges,
            unsigned preferredAxis, bool diagnose = true);

mlir::FailureOr<SourcePlan>
analyzeSource(mlir::Value source, unsigned reductionAxis, bool diagnose = true);

mlir::FailureOr<ParameterAttr>
parameterForExtent(mlir::func::FuncOp kernel, PhysicalExprAttr extent);

bool hasNonUnitFreeAxis(ReduceOp reduce);

bool hasSelectedSegmentExtent(ReduceOp reduce, mlir::func::FuncOp kernel);

mlir::FailureOr<mlir::Value>
dimensionArgument(mlir::func::FuncOp kernel, int64_t dimension);

bool sameFullOrdinalTraversal(llvm::ArrayRef<MakeRangeOp> ranges);

mlir::FailureOr<SourcePlan>
nestedScalarReductionSource(ReduceOp reduce, mlir::func::FuncOp kernel);

} // namespace intent::gpu::reduction
#endif
