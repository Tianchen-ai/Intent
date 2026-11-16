#ifndef INTENT_Block_IR_OPERATIONS_VERIFICATION_H
#define INTENT_Block_IR_OPERATIONS_VERIFICATION_H
#include "Intent/Dialect/Block/IR/BlockOps.h"
namespace intent::block::operation_detail {
mlir::Type elementType(mlir::Type type);
bool sameShape(mlir::Type lhs, mlir::Type rhs);
mlir::LogicalResult verifyHelperRegion(mlir::Operation *owner,
    mlir::Region &region, mlir::TypeRange arguments, mlir::TypeRange results);
}
#endif
