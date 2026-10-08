#ifndef INTENT_INTERFACES_STRUCTUREDOPINTERFACE_H
#define INTENT_INTERFACES_STRUCTUREDOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {

enum class StructuredOpKind { Reduce, Scan, RegionFold, RegionScan };

/// A relation between current SSA values. Only Capture forwards an unchanged
/// value. SameSchema ties accumulator/state shapes, not their runtime contents.
/// Accumulator permits the layer's scalar/slice-to-full-result lifting.
/// SourceSlice, Reduction, Prefix and Emission retain their member semantics;
/// a consumer must not treat all edges as type equality.
enum class StructuredRelationKind {
  Capture, SourceSlice, SameSchema, Accumulator, Reduction, Prefix, Emission
};

struct StructuredValueRelation {
  mlir::Value from;
  mlir::Value to;
  StructuredRelationKind kind;
  /// Source axes for slicing, reduction and prefix relations. Emitted output
  /// axes are derived from the dialect's member relation, not the source rank.
  llvm::SmallVector<int64_t, 2> axes;
};

} // namespace intent

#include "Intent/Interfaces/StructuredOpInterface.h.inc"

namespace intent {

/// Protect all region range queries before type-specific operation verification.
/// ODS validates operand/result segments; this checks block and helper arities.
mlir::LogicalResult verifyStructuredArity(StructuredOpInterface operation);

} // namespace intent

#endif
