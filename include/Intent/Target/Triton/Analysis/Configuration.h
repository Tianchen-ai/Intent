#ifndef INTENT_TARGET_TRITON_ANALYSIS_CONFIGURATION_H
#define INTENT_TARGET_TRITON_ANALYSIS_CONFIGURATION_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <optional>

namespace intent::triton {

inline constexpr int64_t maxTritonTensorElements = 1048576;

std::optional<int64_t> evaluateCompileTimeExpression(
    gpu::PhysicalExprAttr expression, mlir::DictionaryAttr bindings = {});
bool isTritonExpression(gpu::PhysicalExprAttr expression);
bool isTritonFragmentExtent(mlir::Attribute attribute);
std::optional<int64_t> descriptorElementBytes(mlir::Type type);

// Read current native operations and declarations at each complete boundary.
mlir::FailureOr<llvm::SmallVector<gpu::ConfigurationRequirementAttr>>
collectConfigurationRequirements(mlir::func::FuncOp kernel);
mlir::LogicalResult
verifyConfigurationRequirements(mlir::func::FuncOp kernel);

} // namespace intent::triton

#endif
