#ifndef INTENT_Block_TRANSFORMS_PREDICATIONDETAIL_H
#define INTENT_Block_TRANSFORMS_PREDICATIONDETAIL_H

#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "mlir/IR/Builders.h"

namespace intent::block::predication {

bool isFloatToIntegerCast(CastOp operation);
mlir::FailureOr<mlir::Value> selectScalarProduct(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value condition,
    mlir::Value lhs, mlir::Value rhs);
mlir::Value anyActiveLane(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value predicate);

} // namespace intent::block::predication
#endif
