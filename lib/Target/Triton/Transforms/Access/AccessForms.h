#ifndef INTENT_TARGET_TRITON_TRANSFORMS_ACCESS_ACCESSFORMS_H
#define INTENT_TARGET_TRITON_TRANSFORMS_ACCESS_ACCESSFORMS_H

#include "../Configuration/Configurations.h"

namespace intent::triton::detail {

mlir::FailureOr<TensorDescriptorChoiceOp> materializeTensorDescriptorForms(
    mlir::func::FuncOp kernel, llvm::ArrayRef<TritonLocalOptions> localOptions);
mlir::LogicalResult orientPointerLoads(mlir::func::FuncOp kernel);

} // namespace intent::triton::detail

#endif
