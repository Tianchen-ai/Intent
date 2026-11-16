#ifndef INTENT_TARGET_NVIDIA_IR_TARGET_H
#define INTENT_TARGET_NVIDIA_IR_TARGET_H
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "Intent/Target/NVIDIA/IR/TargetDialect.h.inc"
#define GET_ATTRDEF_CLASSES
#include "Intent/Target/NVIDIA/IR/TargetAttrs.h.inc"

namespace intent::nvidia {
DeviceAttr deviceOf(mlir::Operation *operation);
}
#endif
