#ifndef INTENT_TARGET_TRITON_ANALYSIS_CONFIGURATION_H
#define INTENT_TARGET_TRITON_ANALYSIS_CONFIGURATION_H

#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <optional>

namespace intent::triton {

inline constexpr int64_t maxTritonTensorElements = 1048576;

std::optional<int64_t> evaluateCompileTimeExpression(
    block::PhysicalExprAttr expression, mlir::DictionaryAttr bindings = {});
bool isTritonExpression(block::PhysicalExprAttr expression);
bool isTritonFragmentExtent(mlir::Attribute attribute);
std::optional<int64_t> descriptorElementBytes(mlir::Type type);

// Read current native operations and declarations at each complete boundary.
mlir::FailureOr<llvm::SmallVector<block::ConfigurationRequirementAttr>>
collectConfigurationRequirements(mlir::func::FuncOp kernel);
mlir::LogicalResult
verifyConfigurationRequirements(mlir::func::FuncOp kernel);

} // namespace intent::triton

#endif
