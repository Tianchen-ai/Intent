#ifndef INTENT_DIALECT_GPU_TRANSFORMS_RESOURCES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_RESOURCES_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

mlir::FailureOr<llvm::SmallVector<mlir::DictionaryAttr>> filterConfigurationRequirements(
    mlir::func::FuncOp kernel, llvm::ArrayRef<mlir::DictionaryAttr> rows,
    llvm::ArrayRef<ConfigurationRequirementAttr> requirements);

} // namespace intent::gpu

#endif
