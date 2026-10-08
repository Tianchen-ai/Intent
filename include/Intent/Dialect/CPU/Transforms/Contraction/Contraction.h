#ifndef INTENT_DIALECT_CPU_TRANSFORMS_CONTRACTION_CONTRACTION_H
#define INTENT_DIALECT_CPU_TRANSFORMS_CONTRACTION_CONTRACTION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cpu {

struct Configuration;
class ImplementationRegistry;

mlir::LogicalResult normalizeContractionSources(mlir::func::FuncOp function);
mlir::LogicalResult normalizeContractions(mlir::func::FuncOp function);
// Region instantiation exposes concrete storage relationships. Fold only
// input descriptors here, preserving each selected matrix implementation.
mlir::LogicalResult foldContractionInputs(
    mlir::func::FuncOp function, const ImplementationRegistry &implementations);
mlir::LogicalResult blockContractions(
    mlir::func::FuncOp function, const Configuration &configuration,
    const ImplementationRegistry &implementations);

} // namespace intent::cpu
#endif
