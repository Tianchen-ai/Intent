#ifndef INTENT_COMPILER_REGISTRATION_H
#define INTENT_COMPILER_REGISTRATION_H

#include "mlir/IR/DialectRegistry.h"

namespace intent::compiler {
/// The same dialects, provider interfaces and pass inventory serve both tools
/// and embedded callers. Registration does not discover hardware or SDKs.
void registerDialects(mlir::DialectRegistry &registry);
void registerPasses();
} // namespace intent::compiler
#endif
