#ifndef INTENT_DIALECT_Block_IR_ACCESSOPINTERFACE_H
#define INTENT_DIALECT_Block_IR_ACCESSOPINTERFACE_H

#include "Intent/Dialect/Block/IR/BlockTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include <optional>

namespace intent::block {

// These kinds describe the operation's existing semantics. They do not grant
// permission to predicate, reorder, duplicate, or alias an access.
// Payload order for compare-exchange is expected, desired; its predicate follows
// that payload schema rather than the {old_value, success} result record.
enum class AccessKind {
  Load, Gather, Store, ScatterReduce,
  AtomicLoad, AtomicStore, AtomicRMW, AtomicCompareExchange
};

class AccessOpInterface;
mlir::LogicalResult verifyAccessSchema(AccessOpInterface access);

// Coordinate slots retain their operand occurrence order. source_axes maps
// those slots to resource axes; reordering a resource's coordinates must carry
// these projections with them, not reinterpret Cartesian result occurrences.
// Axis queries permit unresolved physical extents and retain known pairs when
// the result is Unknown. Such partial relations are useful to refine schemas;
// only an Exact coordinate projection may be used to lower an access.
BroadcastProjection queryAccessCoordinateAxes(AccessOpInterface access,
                                              unsigned coordinateIndex);
BroadcastProjection queryAccessCoordinateProjection(AccessOpInterface access,
                                                    unsigned coordinateIndex);

} // namespace intent::block

#include "Intent/Dialect/Block/IR/AccessOpInterface.h.inc"

#endif
