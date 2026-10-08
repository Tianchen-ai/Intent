#ifndef INTENT_CONVERSION_INDEXEDACCESS_H
#define INTENT_CONVERSION_INDEXEDACCESS_H

#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Conversion/LogicalShape.h"
#include "mlir/IR/ValueRange.h"
#include <functional>

namespace intent {

// Logical coordinates and member count, never a physical tile capacity.
struct IndexRange {
  mlir::Value begin;
  mlir::Value end;
  mlir::Value step;
  mlir::Value count;
};

// One immediately reified canonical term. New-axis terms have no source
// coordinate; otherwise exactly one of these alternatives is present. The
// original IndexTermFact owns source/result-axis occurrence and provenance.
struct ReifiedIndexTerm {
  mlir::Value coordinate;
  mlir::Value tensor;
  std::optional<IndexRange> range;
};

struct IndexTermMaterialization {
  std::function<mlir::Value(int64_t)> constant;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value)> scalar;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, unsigned)> extent;
  std::function<mlir::FailureOr<IndexRange>(mlir::Value)> domain;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, mlir::Value)> add;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, mlir::Value)> subtract;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, mlir::Value)> multiply;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, mlir::Value)> ceilDivide;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, mlir::Value)> maximum;
};

mlir::FailureOr<ReifiedIndexTerm> materializeIndexTerm(
    mlir::Value source, const IndexTermFact &term,
    const IndexTermMaterialization &materialization);

// The members are logical result ordinals. Callers invoke this inside the
// active read branch. The extractor receives canonical tensor SSA and its
// logical coordinates, and owns only the physical read and index conversion.
mlir::FailureOr<llvm::SmallVector<mlir::Value>> materializeIndexCoordinates(
    const IndexRelationFact &relation, mlir::ValueRange members,
    const IndexTermMaterialization &materialization,
    llvm::function_ref<mlir::FailureOr<mlir::Value>(mlir::Value,
                                                  mlir::ValueRange)>
        extractIndex);

// LogicalShape's existing SSA/host-expression adapters use the same slice
// defaults and length arithmetic as the access reifier.
mlir::FailureOr<mlir::OpFoldResult> reifyIndexSliceExtent(
    CanonicalKernelAnalysis &analysis, mlir::Value source,
    const IndexTermFact &term, const LogicalShapeReification &reification);

} // namespace intent

#endif
