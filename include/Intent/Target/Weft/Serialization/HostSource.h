#ifndef INTENT_TARGET_WEFT_SERIALIZATION_HOSTSOURCE_H
#define INTENT_TARGET_WEFT_SERIALIZATION_HOSTSOURCE_H

#include "Intent/Serialization/NativeSource.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent::weft_provider {

// The host surface owns C spelling and its matching legality checks. It borrows
// ranked-memory metadata and structured control from the common native emitter.
class HostSourceEmitter : public NativeSourceEmitter {
public:
  HostSourceEmitter(mlir::ModuleOp module, llvm::raw_ostream &output);

  std::string nativeType(mlir::Type type) override;
  std::string offsetPointer(llvm::StringRef base,
                            llvm::StringRef offset) override;
  std::string pointerAsIndex(llvm::StringRef base) override;
  mlir::LogicalResult emitNativeOperation(mlir::Operation *operation) override;

  static mlir::LogicalResult verifyOperation(mlir::Operation *operation);

private:
  static const OperationEmitters<HostSourceEmitter> &emitters();
  mlir::LogicalResult emitAllocation(mlir::Operation *operation);
  mlir::LogicalResult emitParallel(mlir::Operation *operation);

  mlir::ModuleOp module;
};

mlir::LogicalResult verifyHostSourceProgram(mlir::ModuleOp module);

} // namespace intent::weft_provider
#endif
