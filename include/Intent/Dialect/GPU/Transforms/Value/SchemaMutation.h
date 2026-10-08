#ifndef INTENT_DIALECT_GPU_TRANSFORMS_SCHEMAMUTATION_H
#define INTENT_DIALECT_GPU_TRANSFORMS_SCHEMAMUTATION_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {

using ValueTypeChangeCallback = llvm::function_ref<void(mlir::Value, mlir::Type)>;

void setPhysicalValueType(mlir::Value value, mlir::Type type,
                          ValueTypeChangeCallback changed = {});

bool hasSchemaBoundary(mlir::Operation *operation);

// Cloning selects all control results/formals before rebuilding their users.
// Reject conflicting selections for one exact control slot group. Projecting
// incoming values afterwards must not change that selected signature.
mlir::LogicalResult verifySelectedControlSchemas(mlir::Operation *operation);
mlir::LogicalResult projectSelectedControlSchemas(
    mlir::Operation *operation, mlir::OpBuilder::Listener *listener = nullptr);

// Project incoming operand slots at their own boundary. Shared SSA seeds are
// never rewritten as if distinct carried components were one schema identity.
mlir::LogicalResult closeSchemaBoundary(
    mlir::Operation *operation, ValueTypeChangeCallback changed = {},
    mlir::OpBuilder::Listener *listener = nullptr);

// Apply an explicit extent decision to the selected control component's
// incoming slots. This never propagates that decision into a shared seed SSA.
mlir::LogicalResult projectSchemaBoundary(
    mlir::Value target, ValueTypeChangeCallback changed = {},
    mlir::OpBuilder::Listener *listener = nullptr);

} // namespace intent::gpu
#endif
