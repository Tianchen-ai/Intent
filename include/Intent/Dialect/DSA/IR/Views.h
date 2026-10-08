#ifndef INTENT_DIALECT_DSA_IR_VIEWS_H
#define INTENT_DIALECT_DSA_IR_VIEWS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace intent::dsa {

// Addressable numerical elements. Native arithmetic primitives impose their
// own, narrower operand requirements independently of storage legality.
bool isStorageElementType(mlir::Type type);

// A complete, zero-offset contiguous local view rooted in an allocation or a
// borrowed collective argument. Partial or differently strided aliases fail.
bool isCompleteLocalStorageView(mlir::Value value);

// A static, bounded contiguous alias of complete local storage. This is view
// legality only; it does not grant whole-storage coverage to a reader/writer.
bool isBoundedContiguousLocalView(mlir::Value value);

// Exact whole-storage coverage, not merely a common allocation origin. This
// accepts identity descriptor casts and complete contiguous reinterpretations.
bool isCompleteStorageViewOf(mlir::Value value, mlir::Value origin);

// Re-expose that storage at its selected capacity rank, reusing an existing
// matching view or allocation whenever possible. Ownership and elements do not
// change, and this builder never creates a partial view.
mlir::FailureOr<mlir::Value>
materializeCollectiveView(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value source, mlir::MemRefType requested);

} // namespace intent::dsa

#endif
