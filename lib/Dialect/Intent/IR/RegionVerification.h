#ifndef INTENT_DIALECT_INTENT_IR_REGIONVERIFICATION_H
#define INTENT_DIALECT_INTENT_IR_REGIONVERIFICATION_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"

namespace intent::detail {

mlir::LogicalResult verifyPureYieldSchema(mlir::Operation *owner,
                                        mlir::Region &region,
                                        mlir::TypeRange arguments,
                                        mlir::TypeRange yielded);

} // namespace intent::detail

#endif
