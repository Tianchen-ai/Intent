#ifndef INTENT_DIALECT_GPU_IR_MEMORYEFFECTS_H
#define INTENT_DIALECT_GPU_IR_MEMORYEFFECTS_H

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace intent::gpu {

// Establishes knowledge in the current control domain without modifying an
// addressable allocation. As with LLVM's inaccessible-memory assume effect,
// this keeps the assumption available to later analyses without claiming a
// write to the resource whose bounds it describes.
struct AssumptionResource
    : mlir::SideEffects::Resource::Base<AssumptionResource> {
  llvm::StringRef getName() final { return "GPUAssumptions"; }
};

} // namespace intent::gpu

#endif
