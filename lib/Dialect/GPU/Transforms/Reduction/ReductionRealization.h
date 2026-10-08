#ifndef INTENT_GPU_TRANSFORMS_REDUCTIONREALIZATION_H
#define INTENT_GPU_TRANSFORMS_REDUCTIONREALIZATION_H

#include "ReductionAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::gpu::reduction {

// Private realization entry points. The public pass driver owns their order
// and refreshes worklists after rewrites; these are not independently run passes.

mlir::FailureOr<bool> realizeStaticPaddingReduce(
    ReduceOp reduce, mlir::func::FuncOp kernel);

mlir::LogicalResult neutralizeReductionTails(
    ReduceOp reduce, mlir::func::FuncOp kernel);

mlir::FailureOr<bool> realizeFullCoverageReduce(
    ReduceOp reduce, mlir::func::FuncOp kernel);

mlir::LogicalResult decomposeMultiAxisReduce(
    ReduceOp reduce, mlir::func::FuncOp kernel);

enum class ReductionCarryForm { Automatic, CompactSummary };

mlir::LogicalResult realizeRuntimeReduce(
    ReduceOp reduce, llvm::ArrayRef<SourcePlan> sourcePlans,
    mlir::func::FuncOp kernel, bool tileProducerFreeAxis,
    ReductionCarryForm carryForm = ReductionCarryForm::Automatic);

bool hoistNestedReduction(mlir::scf::ForOp outer, mlir::func::FuncOp kernel);

bool sinkReductionIntoSourceIf(ReduceOp reduce, mlir::func::FuncOp kernel);

// Collapse sole-use ordinary reduction chains over complete physical members.
bool normalizeCompletedReductions(mlir::func::FuncOp kernel);

} // namespace intent::gpu::reduction
#endif
