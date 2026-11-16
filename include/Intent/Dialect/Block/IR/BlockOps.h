#ifndef INTENT_DIALECT_Block_IR_BlockOPS_H
#define INTENT_DIALECT_Block_IR_BlockOPS_H

#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "Intent/Dialect/Block/IR/BlockTypes.h"
#include "Intent/Dialect/Block/IR/AccessOpInterface.h"
#include "Intent/Dialect/Block/IR/FragmentOpInterface.h"
#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define GET_OP_CLASSES
#include "Intent/Dialect/Block/IR/BlockOps.h.inc"

namespace intent::block {

struct DecodedCoordinate {
  mlir::Value linear;
  mlir::ValueRange extents;
  unsigned axis;
};
std::optional<DecodedCoordinate> queryDecodedCoordinate(mlir::Value value);

mlir::FailureOr<mlir::ArrayAttr>
inferReshapeReassociation(FragmentType source, FragmentType result,
                          unsigned sourcePrefix = 0,
                          unsigned resultPrefix = 0);
mlir::FailureOr<mlir::ArrayAttr>
inferReshapeReassociation(mlir::MLIRContext *context,
                          llvm::ArrayRef<mlir::Attribute> sourceShape,
                          llvm::ArrayRef<mlir::Attribute> resultShape);

} // namespace intent::block

#endif
