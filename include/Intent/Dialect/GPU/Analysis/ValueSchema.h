#ifndef INTENT_DIALECT_GPU_ANALYSIS_VALUESCHEMA_H
#define INTENT_DIALECT_GPU_ANALYSIS_VALUESCHEMA_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::gpu {

// One positional summary/state component. Seeds are kept as operand slots:
// different components may intentionally start from the same SSA constant.
struct StructuredSchemaGroup {
  mlir::OpOperand *producer;
  unsigned seedOperand;
  llvm::SmallVector<mlir::BlockArgument> arguments;
  llvm::SmallVector<mlir::Value> results;
  llvm::SmallVector<mlir::OpOperand *> yields;
};
llvm::SmallVector<StructuredSchemaGroup>
queryStructuredSchemaGroups(mlir::Operation *operation);

// A scan assembles each emitted fragment in the same axis order as its slice.
// The member axis is owned by the current source-slice relation; it is not a
// free-axis extent equality. Record fields retain their positional paths.
mlir::FailureOr<unsigned> queryRegionScanEmissionAxis(
    RegionScanOp scan, unsigned output,
    llvm::ArrayRef<unsigned> fieldPath = {});

// These queries inspect current IR and return facts/types without changing it.
bool isShapeBound(PhysicalExprAttr bound);
mlir::Type scalarCallbackType(mlir::Type type);
bool canPredicateValueOperation(mlir::Operation *operation);
bool variesWithIteration(mlir::Value value, mlir::scf::ForOp loop,
                         llvm::DenseMap<mlir::Value, bool> &known);
std::pair<uint64_t, int64_t>
nextPhysicalAxisIdentities(mlir::func::FuncOp kernel);
PhysicalExprAttr multiplyExtent(PhysicalExprAttr lhs, PhysicalExprAttr rhs);
PhysicalExprAttr productExtent(mlir::MLIRContext *context,
                               llvm::ArrayRef<mlir::Attribute> shape,
                               llvm::ArrayRef<int64_t> axes, unsigned prefix);
bool isIntroducedReshapeUnitAxis(mlir::Value value, unsigned fragmentAxis);
std::optional<unsigned> reshapeInputAxis(ReshapeOp reshape, unsigned resultAxis);
mlir::Value stripAdditiveProjection(mlir::Value value, bool singleUse = false);
bool isLiteralZeroProjection(mlir::Value value);
mlir::FailureOr<FragmentType> queryValueSchema(
    mlir::func::FuncOp kernel, FragmentType target,
    mlir::ValueRange contributors);
mlir::FailureOr<FragmentType> queryAccessResultSchema(
    mlir::func::FuncOp kernel, AccessOpInterface access);
mlir::FailureOr<uint64_t> blockedDimension(mlir::Attribute attribute);
bool hasBlockedDimension(mlir::func::FuncOp kernel, uint64_t dimension);

} // namespace intent::gpu

#endif
