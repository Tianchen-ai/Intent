#ifndef INTENT_DIALECT_GPU_ANALYSIS_HELPERS_H
#define INTENT_DIALECT_GPU_ANALYSIS_HELPERS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include <array>
#include <optional>
#include <string>

namespace intent::gpu {

// Numerical lane semantics and coordinate relations are separate contracts.
// This requires scalarizable elementwise semantics as well as the operation's
// current fragment relation; a pure shape-producing operation is not sufficient.
bool isLaneWisePointwiseOperation(mlir::Operation *operation);
// A projection that forwards the same scalar lane in row-major order. Unit
// dimensions may be inserted/removed; replication of a varying lane may not.
mlir::Value laneWiseProjectionSource(mlir::Operation *operation);
bool isLaneWiseValueOperation(mlir::Operation *operation);

// Current, closed helper only. Explicit capture formals are ordinary inputs;
// implicit captures, effects and nested control/collectives are not lane-wise.
// Does not prove associativity, commutativity, identity or provider capability.
mlir::LogicalResult proveLaneWiseHelper(mlir::Region &region,
                                      std::string *reason = nullptr);

struct BinaryCombine {
  BinaryOp operation;
  // For each actual BinaryOp operand, the corresponding helper formal slot.
  // A reversed noncommutative combine remains reversed, never normalized.
  std::array<unsigned, 2> arguments;

  BinaryOperator kind() const {
    auto current = operation;
    return current.getOperatorKind();
  }
};

std::optional<BinaryCombine> queryBinaryCombine(mlir::Region &region);

} // namespace intent::gpu
#endif
