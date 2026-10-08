#ifndef INTENT_DIALECT_CPU_IR_IMPLEMENTATIONPROVIDER_H
#define INTENT_DIALECT_CPU_IR_IMPLEMENTATIONPROVIDER_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace intent::cpu {

class ImplementationRegistry;

/// Compiler initialization installs the providers compiled into the embedding
/// application. Registries are immutable and owned by this context's dialect;
/// selected implementation bindings remain in the current CPU program.
class ImplementationProviderInterface
    : public mlir::DialectInterface::Base<ImplementationProviderInterface> {
public:
  explicit ImplementationProviderInterface(mlir::Dialect *dialect) : Base(dialect) {}
  virtual mlir::FailureOr<const ImplementationRegistry *>
  lookup(llvm::StringRef provider) const = 0;
};

} // namespace intent::cpu

#endif
