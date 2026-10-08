#ifndef INTENT_DIALECT_GPU_IR_GPUATTRS_H
#define INTENT_DIALECT_GPU_IR_GPUATTRS_H

#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Attributes.h"

namespace mlir { class Operation; }

#include "Intent/Dialect/GPU/IR/GPUAttrsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/GPU/IR/GPUAttrs.h.inc"

namespace intent::gpu {
bool isCompileTimePhysicalExpr(PhysicalExprAttr expression);
ParameterAttr lookupParameterDeclaration(mlir::Operation *anchor,
                                         ParameterRefAttr reference);
mlir::LogicalResult verifyParameterDeclarations(mlir::Operation *kernel);
}

#endif
