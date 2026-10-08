#ifndef INTENT_DIALECT_DSA_IR_MEMORYEFFECTS_H
#define INTENT_DIALECT_DSA_IR_MEMORYEFFECTS_H

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace intent {
struct BufferStoragePolicy;
}

namespace intent::dsa {

// Transfer/order state is distinct from the contents of any buffer. Its effects
// keep explicit waits and asynchronous issue points visible to standard passes.
struct TransferOrderResource
    : mlir::SideEffects::Resource::Base<TransferOrderResource> {
  llvm::StringRef getName() final { return "IntentDSATransferOrder"; }
};

enum class CompletionScope { Immediate, WorkUnit, ExecutionGroup };

CompletionScope requiredCompletion(mlir::Operation *operation);
bool completesTransfers(mlir::Operation *operation, CompletionScope scope);
bool hasStorageOrdering(mlir::Operation *operation);
intent::BufferStoragePolicy storagePolicy();

} // namespace intent::dsa
#endif
