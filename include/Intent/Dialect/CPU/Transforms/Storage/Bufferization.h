#ifndef INTENT_DIALECT_CPU_TRANSFORMS_BUFFERIZATION_H
#define INTENT_DIALECT_CPU_TRANSFORMS_BUFFERIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace intent::cpu {

void registerValueBufferizationInterfaces(mlir::DialectRegistry &registry);

// Close tensor SSA, storage decisions and ownership in the same transformation.
// Conditional releases remain explicit bufferization.dealloc operations until
// provider legalization lowers the already selected ownership control flow.
mlir::LogicalResult bufferizeValues(mlir::ModuleOp module);
mlir::LogicalResult lowerOwnership(mlir::ModuleOp module);

} // namespace intent::cpu

#endif
