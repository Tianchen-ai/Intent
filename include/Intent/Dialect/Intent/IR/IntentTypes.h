#ifndef INTENT_DIALECT_INTENT_IR_INTENTTYPES_H
#define INTENT_DIALECT_INTENT_IR_INTENTTYPES_H

#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Diagnostics.h"

#define GET_TYPEDEF_CLASSES
#include "Intent/Dialect/Intent/IR/IntentTypes.h.inc"

namespace intent {
bool isCanonicalScalarType(mlir::Type type);
mlir::LogicalResult verifyCanonicalType(
    llvm::function_ref<mlir::InFlightDiagnostic()> error, mlir::Type type);
mlir::LogicalResult verifyCanonicalType(mlir::Operation *owner, mlir::Type type);
}

#endif
