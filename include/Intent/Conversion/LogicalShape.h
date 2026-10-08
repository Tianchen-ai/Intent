#ifndef INTENT_CONVERSION_LOGICALSHAPE_H
#define INTENT_CONVERSION_LOGICALSHAPE_H

#include "Intent/Analysis/CanonicalKernel.h"
#include <functional>

namespace intent {

// OpFoldResult lets the same relation reifier produce either SSA or an existing
// typed host expression. It does not broaden launch visibility.
struct LogicalShapeReification {
  std::function<mlir::OpFoldResult(mlir::Value, llvm::ArrayRef<unsigned>, unsigned)>
      lookup;
  std::function<mlir::FailureOr<mlir::OpFoldResult>(const TensorExtentFact &)> leaf;
  std::function<mlir::FailureOr<mlir::OpFoldResult>(mlir::OpFoldResult,
                                                 mlir::OpFoldResult)> multiply;
  std::function<mlir::FailureOr<mlir::OpFoldResult>(mlir::OpFoldResult,
                                                 mlir::OpFoldResult)> exactDivide;
  std::function<mlir::FailureOr<mlir::OpFoldResult>(mlir::OpFoldResult,
                                                 mlir::OpFoldResult)> subtract;
  std::function<mlir::FailureOr<mlir::OpFoldResult>(mlir::OpFoldResult,
                                                 mlir::OpFoldResult)> ceilDivide;
  std::function<mlir::FailureOr<mlir::OpFoldResult>(mlir::OpFoldResult,
                                                 mlir::OpFoldResult)> maximum;
};

mlir::FailureOr<mlir::OpFoldResult> reifyLogicalExtent(
    CanonicalKernelAnalysis &analysis, mlir::Value value, unsigned axis,
    const LogicalShapeReification &reification,
    llvm::ArrayRef<unsigned> fieldPath = {});

// Family bindings describe actual values at the current insertion scope. They
// never replace logical dimensions with a physical tile's capacity.
struct LogicalShapeMaterialization {
  std::function<mlir::Value(mlir::Value, llvm::ArrayRef<unsigned>, unsigned)>
      lookup;
  std::function<mlir::FailureOr<mlir::Value>(const TensorExtentFact &)> leaf;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, mlir::Value)> multiply;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, mlir::Value)> exactDivide;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, mlir::Value)> subtract;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, mlir::Value)> ceilDivide;
  std::function<mlir::FailureOr<mlir::Value>(mlir::Value, mlir::Value)> maximum;
};

// The SSA convenience adapter for the same reifier. Inferred reshape
// arithmetic is interpreted once here; the family owns integer spelling and
// actual descriptor/domain/formal bindings. The canonical KIR stays immutable.
mlir::FailureOr<mlir::Value> materializeLogicalExtent(
    CanonicalKernelAnalysis &analysis, mlir::Value value, unsigned axis,
    const LogicalShapeMaterialization &materialization,
    llvm::ArrayRef<unsigned> fieldPath = {});

} // namespace intent

#endif
