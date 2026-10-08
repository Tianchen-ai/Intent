#ifndef INTENT_DIALECT_GPU_ANALYSIS_CONFIGURATIONEXPRESSIONS_H
#define INTENT_DIALECT_GPU_ANALYSIS_CONFIGURATIONEXPRESSIONS_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

// Substitute one configuration's bindings and fold checked concrete arithmetic.
// ABI leaves and parameters absent from the row remain symbolic. Invalid
// concrete arithmetic or a malformed binding returns failure.
mlir::FailureOr<PhysicalExprAttr>
instantiateConfigurationExpression(PhysicalExprAttr expression,
                                   mlir::DictionaryAttr bindings);

// Prove the inequality for every current Shared or Complete tuple, retaining
// correlations within each row. Runtime leaves retain their checked-expression
// domains; shared signed-order rules prove the remaining symbolic relations.
// A tuple-dependent proof requires nonempty configurations and every comparison
// to succeed; missing bindings never acquire default values.
// Shared ResidentWorkers remains symbolic until its provider binding is final.
bool configurationExpressionAtMost(mlir::func::FuncOp kernel,
                                   PhysicalExprAttr lhs,
                                   PhysicalExprAttr rhs);
bool configurationExpressionLessThan(mlir::func::FuncOp kernel,
                                     PhysicalExprAttr lhs,
                                     PhysicalExprAttr rhs);

} // namespace intent::gpu

#endif
