#ifndef INTENT_DIALECT_CPU_TRANSFORMS_STRUCTURE_PRODUCERREPLAY_H
#define INTENT_DIALECT_CPU_TRANSFORMS_STRUCTURE_PRODUCERREPLAY_H

#include "Intent/Dialect/CPU/Analysis/ProducerReplay.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace intent::cpu {
/// Clone precisely the analyzed payload, using the caller's concrete frontier.
/// The snapshot must have passed canReplayProducerAt before target IR is built.
/// Nested regions are cloned whole, preserving conditional reads and yields.
mlir::LogicalResult cloneProducerPayload(const ProducerReplay &payload,
                                         mlir::OpBuilder &builder,
                                         mlir::IRMapping &mapping);

/// Scalar materialization for adapters that replay selected subexpressions
/// (e.g. invariant operands of an otherwise vectorized consumer).
mlir::FailureOr<mlir::Value> materializeProducerValue(
    const ProducerReplay &payload, mlir::Value value, mlir::OpBuilder &builder,
    mlir::IRMapping &mapping);
} // namespace intent::cpu

#endif
