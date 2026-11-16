#ifndef INTENT_DIALECT_Block_TRANSFORMS_STORAGE_WORKSPACE_H
#define INTENT_DIALECT_Block_TRANSFORMS_STORAGE_WORKSPACE_H

#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent::block {

// Materialization creates resources in the current program. Only the terminal
// workspace lowering adds invocation arguments and allocation layout.
bool isProgramAllocationContext(mlir::Operation *operation,
                                 mlir::func::FuncOp kernel);
BufferOp createProgramBuffer(mlir::OpBuilder &builder, mlir::Location location,
                             mlir::Type elementType, mlir::ArrayAttr shape,
                             uint64_t owner);
BufferOp createInvocationBuffer(mlir::func::FuncOp kernel,
                                mlir::Location location,
                                mlir::Type elementType, mlir::ArrayAttr shape,
                                uint64_t owner);
mlir::LogicalResult lowerWorkspaceAllocations(mlir::ModuleOp module);

} // namespace intent::block
#endif
