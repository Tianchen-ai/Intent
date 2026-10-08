#ifndef INTENT_TARGET_TRITON_TRANSFORMS_CONFIGURATIONS_H
#define INTENT_TARGET_TRITON_TRANSFORMS_CONFIGURATIONS_H
#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"
#include "Intent/Target/Triton/Analysis/Configuration.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <optional>
namespace intent::triton {
struct TritonLocalOptions {
  int64_t warps;
  int64_t stages;
  int64_t ctas;
};


mlir::FailureOr<llvm::SmallVector<TritonLocalOptions>> declareProviderOptions(
    mlir::func::FuncOp kernel,
    bool requiresCtaSynchronization, llvm::ArrayRef<mlir::scf::ForOp> loadPipelineLoops);
mlir::LogicalResult materializeLegalConfigs(
    mlir::func::FuncOp kernel, TensorDescriptorChoiceOp descriptorChoice);
mlir::LogicalResult finalizeConfigurationRequirements(mlir::func::FuncOp kernel);
} // namespace intent::triton
#endif
