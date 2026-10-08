#ifndef INTENT_CPU_TRANSFORMS_INPUTWINDOWS_H
#define INTENT_CPU_TRANSFORMS_INPUTWINDOWS_H

#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace intent::cpu {

struct InputWindowDependencies {
  llvm::SmallVector<mlir::Operation *> operations;
  llvm::SmallVector<mlir::Value> frontier;
};

// The returned operations are an ordered, speculatable descriptor/coordinate
// slice in scope. This is a dependency query: a caller that moves operations
// must check their blocks and the availability of the explicit frontier and
// outside values at its chosen insertion point.
std::optional<InputWindowDependencies> inputWindowDependencies(
    mlir::ValueRange values, mlir::Operation *scope,
    mlir::ValueRange frontier = {});

struct ConsumerWindow {
  mlir::memref::SubViewOp view;
  unsigned axis;
  bool transposed;
};

struct InputWindowBounds {
  mlir::AffineMap lower, upper;
  mlir::ValueDimList lowerOperands, upperOperands;
};

std::optional<InputWindowBounds> boundInputWindow(
    const ConsumerWindow &window, mlir::Operation *scope,
    const InputRequirement &requirement);
mlir::Value materializeInputWindowBound(
    mlir::OpBuilder &builder, mlir::Location location, mlir::AffineMap bound,
    const mlir::ValueDimList &operands);

std::optional<ConsumerWindow> consumerWindow(
    mlir::Value source, const InputRequirement &requirement,
    mlir::Operation *consumer);
bool hasIndependentWindowCoordinates(mlir::memref::SubViewOp window,
                                     mlir::Operation *loop,
                                     mlir::Value groupCoordinate = {});

} // namespace intent::cpu
#endif
