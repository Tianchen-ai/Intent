#ifndef INTENT_DIALECT_CPU_TRANSFORMS_CONFIGURATION_H
#define INTENT_DIALECT_CPU_TRANSFORMS_CONFIGURATION_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace intent::cpu {

class ImplementationRegistry;

// One profile row. Shared fields bind the task/block program; local fields are
// validated by the provider's family schema and consumed by implementations.
struct Configuration {
  int64_t taskGrain;
  int64_t tileM;
  int64_t tileN;
  int64_t tileK;
  int64_t regionSize;
  mlir::DictionaryAttr local;
  // Profile selection only; executable consumers use per-operation bindings.
  mlir::ArrayAttr implementations;

  int64_t parameter(llvm::StringRef name) const {
    return mlir::cast<mlir::IntegerAttr>(local.get(name)).getInt();
  }
};

mlir::LogicalResult materializeCPUConfigurations(
    mlir::ModuleOp module, const ImplementationRegistry &implementations,
    llvm::StringRef defaults, llvm::StringRef overrides);

} // namespace intent::cpu
#endif
