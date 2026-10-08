#ifndef INTENT_TARGET_CUTILE_ANALYSIS_CONFIGURATION_H
#define INTENT_TARGET_CUTILE_ANALYSIS_CONFIGURATION_H
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intent::cutile {
mlir::FailureOr<llvm::SmallVector<gpu::ConfigurationRequirementAttr>>
collectConfigurationRequirements(mlir::func::FuncOp kernel);
mlir::LogicalResult verifyClosedConfigs(mlir::func::FuncOp kernel);
}
#endif
