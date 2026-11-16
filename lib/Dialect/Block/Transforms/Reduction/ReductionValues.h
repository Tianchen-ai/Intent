#ifndef INTENT_Block_TRANSFORMS_REDUCTIONVALUES_H
#define INTENT_Block_TRANSFORMS_REDUCTIONVALUES_H

#include "Intent/Dialect/Block/IR/BlockOps.h"

#include <string>

namespace intent::block::reduction {

// Local reduction value construction: fragment schemas, range attributes and
// typed combine lifting/inlining shared by the realization paths.

void inheritRangeAuthority(mlir::Operation *target, MakeRangeOp source);

PhysicalExprAttr expression(mlir::MLIRContext *context, PhysicalExprKind kind,
                           int64_t value = 0, llvm::StringRef symbol = {},
                           llvm::ArrayRef<mlir::Attribute> operands = {});

PhysicalExprAttr nextPowerOfTwo(PhysicalExprAttr source);

FragmentType replaceExtent(FragmentType source, unsigned axis,
                           PhysicalExprAttr extent, mlir::Type element = {});

mlir::FailureOr<mlir::Value> predicateForReductionSource(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value predicate,
    FragmentType source, unsigned reductionAxis);

mlir::FailureOr<llvm::SmallVector<mlir::Value>>
inlinePureRegion(mlir::OpBuilder &builder, mlir::Region &region,
                 mlir::ValueRange arguments, std::string &reason);

mlir::Type dataElementType(mlir::Type type);

FragmentType withElementType(FragmentType schema, mlir::Type elementType);

mlir::FailureOr<mlir::Value> alignToExecutionSchema(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    FragmentType executionSchema);

bool prepareVectorAccumulation(ReduceOp reduce,
                               llvm::ArrayRef<FragmentType> accumulatorTypes,
                               mlir::Region &combine);

} // namespace intent::block::reduction
#endif
