#ifndef INTENT_DIALECT_DSA_IR_DSAOPS_H
#define INTENT_DIALECT_DSA_IR_DSAOPS_H
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "Intent/Dialect/DSA/IR/DSADialect.h.inc"
#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/DSA/IR/DSAAttrs.h.inc"
#define GET_OP_CLASSES
#include "Intent/Dialect/DSA/IR/DSAOps.h.inc"
namespace intent::dsa {
inline constexpr llvm::StringLiteral entryRequirementsAttr = "intent_dsa.entry_requirements";
inline constexpr llvm::StringLiteral fullExtentDimensionsAttr = "intent_dsa.full_extent_dimensions";
constexpr int64_t nramSpace = 1;
constexpr int64_t matrixSpace = 2;
constexpr int64_t sharedSpace = 3;
// Collective formals denote borrowed values, not aliases of identity prototypes.
bool isCollectiveBorrowedArgument(mlir::BlockArgument argument);
// Public dimensions required to fit the current configuration's tile capacity.
// An explicit empty list is valid; an absent or malformed requirement is not.
mlir::FailureOr<mlir::DenseI64ArrayAttr>
queryFullExtentRequirements(mlir::func::FuncOp function);
mlir::LogicalResult verifyProgram(mlir::ModuleOp module);
mlir::LogicalResult verifyRealizedProgram(mlir::ModuleOp module);
}
#endif
