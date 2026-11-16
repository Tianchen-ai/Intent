#ifndef INTENT_DIALECT_Block_IR_BlockATTRS_H
#define INTENT_DIALECT_Block_IR_BlockATTRS_H

#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "mlir/IR/Attributes.h"

namespace mlir { class Operation; }

#include "Intent/Dialect/Block/IR/BlockAttrsEnums.h.inc"

#include "Intent/Dialect/Block/IR/Target.h"
#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/Block/IR/BlockAttrs.h.inc"

namespace intent::block {
bool isCompileTimePhysicalExpr(PhysicalExprAttr expression);
ParameterAttr lookupParameterDeclaration(mlir::Operation *anchor,
                                         ParameterRefAttr reference);
mlir::LogicalResult verifyParameterDeclarations(mlir::Operation *kernel);
}

#endif
