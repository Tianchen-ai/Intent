#ifndef INTENT_DIALECT_CPU_TRANSFORMS_TUNINGPROFILES_H
#define INTENT_DIALECT_CPU_TRANSFORMS_TUNINGPROFILES_H

#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"

namespace intent::cpu {
class ImplementationRegistry;

// Validated input rows only. This object does not retain executable choices or
// participate in interpreting a candidate after its bindings enter CPU IR.
class TuningProfiles {
public:
  static mlir::FailureOr<TuningProfiles> read(
      mlir::ModuleOp module, llvm::StringRef defaults, llvm::StringRef overrides,
      const ImplementationRegistry &implementations);
  llvm::ArrayRef<Configuration> get(llvm::StringRef family) const;

private:
  mlir::LogicalResult readFile(mlir::ModuleOp module, llvm::StringRef path,
                               const ImplementationRegistry &implementations);
  llvm::StringMap<llvm::SmallVector<Configuration>> families;
};
} // namespace intent::cpu
#endif
