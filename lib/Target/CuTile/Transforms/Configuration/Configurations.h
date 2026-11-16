#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_CONFIGURATIONS_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_CONFIGURATIONS_H
#include "Intent/Target/CuTile/IR/Configuration.h"
#include "Intent/Target/CuTile/Analysis/Configuration.h"
#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Transforms/Configuration/TuningProfiles.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intent::cutile {
mlir::FailureOr<block::ParameterRefAttr> declareProviderParameter(
    mlir::func::FuncOp kernel, llvm::StringRef name, block::ParameterRole role,
    bool (*isLegal)(int64_t));
mlir::LogicalResult prepareLaunchConfigurations(mlir::func::FuncOp kernel);
mlir::LogicalResult materializeClosedConfigs(mlir::func::FuncOp kernel);
mlir::LogicalResult finalizeConfigurationRequirements(mlir::func::FuncOp kernel);
} // namespace intent::cutile
#endif
