#ifndef INTENT_TARGET_CUTILE_ANALYSIS_CONFIGURATION_H
#define INTENT_TARGET_CUTILE_ANALYSIS_CONFIGURATION_H
#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intent::cutile {
mlir::FailureOr<llvm::SmallVector<block::ConfigurationRequirementAttr>>
collectConfigurationRequirements(mlir::func::FuncOp kernel);
mlir::LogicalResult verifyClosedConfigs(mlir::func::FuncOp kernel);
}
#endif
