#ifndef INTENT_TARGET_WEFT_TRANSFORMS_SCALARVALUES_H
#define INTENT_TARGET_WEFT_TRANSFORMS_SCALARVALUES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"

namespace intent::weft_provider {

mlir::Type nativeScalarType(mlir::Type type);
mlir::Type nativeValueType(mlir::Type element, llvm::ArrayRef<int64_t> shape,
                           llvm::ArrayRef<int64_t> axes);
mlir::FailureOr<mlir::Type> pointwiseValueType(
    mlir::Type element, mlir::ValueRange operands);
mlir::FailureOr<mlir::Value> createBinaryValue(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value lhs,
    mlir::Value rhs, llvm::StringRef kind);

// Decode standard scalar semantics once, then lift that operation over the
// actual native operand axes. Storage, mapping and task control stay with the
// caller; unsupported native numerical operations never change the source rule.
mlir::FailureOr<mlir::Value> lowerScalarValue(
    mlir::OpBuilder &builder, mlir::Operation *operation,
    mlir::ValueRange operands);

} // namespace intent::weft_provider
#endif
