#ifndef INTENT_DIALECT_DSA_IR_MEMORYEFFECTS_H
#define INTENT_DIALECT_DSA_IR_MEMORYEFFECTS_H

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "Intent/Dialect/DSA/IR/ExecutionInterfaces.h"

namespace intent {
struct BufferStoragePolicy;
}

namespace intent::bangc::physical {

bool hasStorageOrdering(mlir::Operation *operation);
intent::BufferStoragePolicy storagePolicy();

} // namespace intent::bangc::physical
#endif
