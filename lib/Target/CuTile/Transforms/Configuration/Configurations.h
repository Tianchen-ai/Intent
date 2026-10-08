#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_CONFIGURATIONS_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_CONFIGURATIONS_H
#include "Intent/Target/CuTile/IR/Configuration.h"
#include "Intent/Target/CuTile/Analysis/Configuration.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intent::cutile {
mlir::FailureOr<gpu::ParameterRefAttr> declareProviderParameter(
    mlir::func::FuncOp kernel, llvm::StringRef name, gpu::ParameterRole role,
    bool (*isLegal)(int64_t));
mlir::LogicalResult prepareLaunchConfigurations(mlir::func::FuncOp kernel);
mlir::LogicalResult materializeClosedConfigs(mlir::func::FuncOp kernel);
mlir::LogicalResult finalizeConfigurationRequirements(mlir::func::FuncOp kernel);
} // namespace intent::cutile
#endif
