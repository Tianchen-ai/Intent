#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <optional>

namespace intent::cpu {

struct ContractionRequirements;

struct ContractionInitialization {
  mlir::Operation *operation;
  mlir::Value value;
  // Shared SSA initializers may already have been observed by another copy.
  bool erasable;
};
std::optional<ContractionInitialization>
findContractionInitialization(mlir::linalg::GenericOp operation);
mlir::Value foldContractionInput(mlir::Value input, mlir::linalg::GenericOp consumer,
    const ContractionRequirements *requirements = nullptr, unsigned operand = 0);

} // namespace intent::cpu
