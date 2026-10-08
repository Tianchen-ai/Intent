#ifndef INTENT_DIALECT_CPU_ANALYSIS_VIEWRELATIONS_H
#define INTENT_DIALECT_CPU_ANALYSIS_VIEWRELATIONS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace intent::cpu {

// A view of the same elements and offset, with permuted axes or inserted /
// removed unit dimensions. Axis positions are occurrences, not storage IDs.
// The source is an actual descriptor SSA value, never an alias-analysis root.
struct ViewAxisProjection {
  mlir::Value source;
  llvm::SmallVector<std::optional<unsigned>> sourceAxes;
};

mlir::FailureOr<ViewAxisProjection>
queryViewAxisProjection(mlir::Value view);

// Prove row-major element strides. The descriptor may have a nonzero offset:
// consumers must pass its actual base + offset, not its allocation origin.
// These queries prove geometry only, not ownership, lifetime, public shape
// binding, native dtype support, disjointness, or byte-size overflow bounds.
// Requery after IR mutation.
bool isContiguousDescriptor(mlir::Value view);

} // namespace intent::cpu

#endif
