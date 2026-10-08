#ifndef INTENT_DIALECT_CPU_ANALYSIS_EXTENTRELATIONS_H
#define INTENT_DIALECT_CPU_ANALYSIS_EXTENTRELATIONS_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace intent::cpu {

// A read-only bound in terms of current SSA values or shaped-value dimensions.
// Its operands, like any analysis result, must be queried again after mutation.
struct ExtentExpression {
  mlir::AffineMap expression;
  mlir::ValueDimList operands;
};

bool haveEqualExtents(const mlir::ValueBoundsConstraintSet::Variable &lhs,
                      const mlir::ValueBoundsConstraintSet::Variable &rhs);
mlir::FailureOr<ExtentExpression>
queryExtent(const mlir::ValueBoundsConstraintSet::Variable &extent);
mlir::FailureOr<ExtentExpression> queryExtent(mlir::OpFoldResult extent);
std::optional<int64_t>
constantExtentUpperBound(const mlir::ValueBoundsConstraintSet::Variable &extent);

// Return an existing scalar size or a constant without following block argument
// bindings. Unlike equality proofs, a rewrite must preserve that SSA's scope.
std::optional<mlir::OpFoldResult> queryExtentValue(mlir::Value memory, int64_t axis);

// Register CPU helper/ABI facts and descriptor models absent from MLIR 20.
// Standard memref and SCF models are registered by the compiler's MLIR registry.
void registerExtentRelations(mlir::DialectRegistry &registry);

} // namespace intent::cpu
#endif
