#ifndef INTENT_TARGET_MOJO_TRANSFORMS_QUANTIZATION_H
#define INTENT_TARGET_MOJO_TRANSFORMS_QUANTIZATION_H

#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"

namespace intent::mojo {

void registerQuantizedImplementations(cpu::ImplementationRegistry &registry);
mlir::LogicalResult materializeQuantizedComputations(
    mlir::func::FuncOp function, const cpu::ImplementationRegistry &registry);

} // namespace intent::mojo
#endif
