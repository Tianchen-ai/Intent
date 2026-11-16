#ifndef INTENT_TARGET_BANGC_SERIALIZATION_SURFACE_H
#define INTENT_TARGET_BANGC_SERIALIZATION_SURFACE_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

namespace intent::bangc {
std::optional<mlir::LogicalResult>
verifyNativeSourceOperation(mlir::Operation *operation);
std::optional<mlir::LogicalResult>
verifyUnboundNativeSourceOperation(mlir::Operation *operation);
} // namespace intent::bangc

#endif
