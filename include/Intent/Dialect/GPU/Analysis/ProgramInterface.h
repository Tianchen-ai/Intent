#ifndef INTENT_DIALECT_GPU_ANALYSIS_PROGRAMINTERFACE_H
#define INTENT_DIALECT_GPU_ANALYSIS_PROGRAMINTERFACE_H

#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "llvm/ADT/DenseMap.h"

namespace intent::gpu {

struct ProgramArgument {
  mlir::BlockArgument value;
  ArgumentBindingAttr binding;
};

// Snapshot of the current signature; rebuild after changing arguments or bindings.
class ProgramInterface {
public:
  static mlir::FailureOr<ProgramInterface> read(mlir::func::FuncOp kernel);
  llvm::ArrayRef<ProgramArgument> arguments() const { return entries; }
  mlir::BlockArgument resolve(ArgumentRefAttr reference) const;
  mlir::BlockArgument resolve(PhysicalExprAttr expression) const;
  mlir::BlockArgument publicArgument(unsigned ordinal) const;
  mlir::BlockArgument dimension(int64_t identity) const;
  mlir::BlockArgument stride(ArgumentRefAttr view, unsigned axis) const;
  intent::PublicParameterAttr publicParameter(ArgumentRefAttr reference) const;
  intent::PublicParameterAttr publicParameter(mlir::Value value) const;
  intent::InterfaceAttr getPublicInterface() const { return interface; }

private:
  intent::InterfaceAttr interface;
  llvm::SmallVector<ProgramArgument> entries;
  llvm::DenseMap<ArgumentRefAttr, mlir::BlockArgument> references;
};

} // namespace intent::gpu
#endif
