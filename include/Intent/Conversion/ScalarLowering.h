#ifndef INTENT_CONVERSION_SCALARLOWERING_H
#define INTENT_CONVERSION_SCALARLOWERING_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent {

// Logical signedness selects the arithmetic operation before physical integer
// carriers become signless. This does not change public interface attributes.
void realizeIntegerStorage(mlir::ModuleOp module);

mlir::FailureOr<mlir::Value> castScalarValue(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    mlir::Type destination, mlir::Type logicalSource = {});

// Operands are already scalarized by the family. Numerical modes requiring
// target primitives and composite math implementations remain with the caller.
mlir::FailureOr<mlir::Value>
lowerScalarOperation(mlir::Operation *operation, mlir::ValueRange operands,
                     mlir::OpBuilder &builder);

} // namespace intent

#endif
