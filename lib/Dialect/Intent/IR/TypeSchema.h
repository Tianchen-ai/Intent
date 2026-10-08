#ifndef INTENT_DIALECT_INTENT_IR_TYPESCHEMA_H
#define INTENT_DIALECT_INTENT_IR_TYPESCHEMA_H

#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace intent::detail {

bool isIntegerLike(mlir::Type type);
mlir::RankedTensorType getTensorSchema(mlir::Type type);
mlir::Type getElementType(mlir::Type type);
std::optional<unsigned> getLogicalRank(mlir::Type type);
std::optional<uint64_t> getCoordinateSource(mlir::Type type);
std::optional<int64_t> getConstantInteger(mlir::Value value);
mlir::FailureOr<llvm::SmallVector<int64_t>>
getIntegerArray(mlir::Operation *operation, llvm::StringRef name);
mlir::DenseI64ArrayAttr getDimensionIDs(mlir::RankedTensorType tensor);
std::optional<int64_t> getDimensionID(mlir::RankedTensorType tensor, unsigned axis);
bool extentValueMatchesAxis(mlir::Value extent, mlir::RankedTensorType tensor,
                           unsigned axis);
bool sameDimension(mlir::RankedTensorType lhs, unsigned lhsAxis,
                   mlir::RankedTensorType rhs, unsigned rhsAxis);
bool sameTensorShape(mlir::RankedTensorType lhs, mlir::RankedTensorType rhs);
bool compatibleElementType(mlir::Type lhs, mlir::Type rhs);
bool sameDataSchema(mlir::Type lhs, mlir::Type rhs, bool compareElements = true);
bool isBooleanData(mlir::Type type);
bool isNumericData(mlir::Type type);
mlir::FailureOr<llvm::SmallVector<unsigned>>
verifyShapeRelation(mlir::Operation *operation, ShapeRelationAttr relation,
                    mlir::RankedTensorType result, mlir::ValueRange operands,
                    bool allowInferred);
mlir::FailureOr<llvm::SmallVector<int64_t>>
getExtentDimensions(mlir::Operation *operation, unsigned rank);
mlir::FailureOr<llvm::SmallVector<int64_t>>
getIterationExtentDimensions(mlir::Value source);
mlir::LogicalResult verifyShapeOperands(
    mlir::Operation *operation, mlir::ValueRange operands,
    llvm::ArrayRef<unsigned> dynamicOperands, unsigned firstShapeOperand);

} // namespace intent::detail

#endif
