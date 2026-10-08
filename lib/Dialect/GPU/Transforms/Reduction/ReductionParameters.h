#ifndef INTENT_GPU_TRANSFORMS_REDUCTIONPARAMETERS_H
#define INTENT_GPU_TRANSFORMS_REDUCTIONPARAMETERS_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

namespace intent::gpu::reduction {

// Select and bind existing physical parameter declarations for reduction
// traversals and free axes. Candidate policy stays with these mutations.

mlir::FailureOr<ParameterAttr> selectReductionChunk(
    mlir::func::FuncOp kernel, PhysicalSourceAxis source, MakeRangeOp range,
    ParameterRole role, unsigned elementBitWidth, llvm::StringRef name,
    llvm::ArrayRef<int64_t> candidates);

mlir::FailureOr<ParameterAttr> selectReductionTraversalChunk(
    ReduceOp reduce, mlir::Value source, unsigned axis, MakeRangeOp range,
    bool tileProducerFreeAxis = false);

mlir::Value parameterValue(mlir::func::FuncOp kernel, ParameterAttr declaration);

mlir::FailureOr<ParameterAttr>
fullCoverageParameter(mlir::func::FuncOp kernel, PhysicalExprAttr extent);

mlir::LogicalResult bindReductionFreeAxes(
    ReduceOp reduce, mlir::func::FuncOp kernel);

} // namespace intent::gpu::reduction
#endif
