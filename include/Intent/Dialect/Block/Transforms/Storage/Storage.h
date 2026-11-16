#ifndef INTENT_DIALECT_Block_TRANSFORMS_STORAGE_H
#define INTENT_DIALECT_Block_TRANSFORMS_STORAGE_H

#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace intent::block {

mlir::FailureOr<mlir::Value> materializeRetainedSlice(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    unsigned axis, PhysicalExprAttr blockedExtent, mlir::Value coordinates,
    mlir::Operation *insertionAnchor, AxisMapAttr resultMapping = {});

} // namespace intent::block

#endif
