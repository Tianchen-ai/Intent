#ifndef INTENT_CPU_TRANSFORMS_STORAGE_COLLECTIVEBUFFERIZATION_H
#define INTENT_CPU_TRANSFORMS_STORAGE_COLLECTIVEBUFFERIZATION_H

#include "mlir/IR/DialectRegistry.h"

namespace intent::cpu {

void registerCollectiveBufferizationInterfaces(mlir::DialectRegistry &registry);

} // namespace intent::cpu
#endif
