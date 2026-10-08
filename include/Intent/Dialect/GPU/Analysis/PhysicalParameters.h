#ifndef INTENT_DIALECT_GPU_ANALYSIS_PHYSICALPARAMETERS_H
#define INTENT_DIALECT_GPU_ANALYSIS_PHYSICALPARAMETERS_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

// A missing table/reference is returned as null, never as an implicit empty
// parameter space. These queries do not cache declaration state.
mlir::ArrayAttr getParameterDeclarations(mlir::func::FuncOp kernel);
ParameterAttr lookupParameter(mlir::func::FuncOp kernel, mlir::StringAttr name);
ParameterAttr lookupParameter(mlir::func::FuncOp kernel, ParameterRefAttr reference);

// Snapshot of kernel-owned declarations, independent of SSA reads. Rebuild
// after any declaration mutation; candidate bindings remain in current IR.
class ParameterSpace {
public:
  static mlir::FailureOr<ParameterSpace> read(mlir::func::FuncOp kernel);
  llvm::ArrayRef<ParameterAttr> declarations() const { return parameters; }
  llvm::ArrayRef<ParameterAttr> extentDeclarations() const { return extents; }
  ParameterAttr lookup(mlir::StringAttr name) const;
  ParameterAttr lookup(ParameterRefAttr reference) const;
  ParameterAttr find(ParameterRole role) const;
  mlir::LogicalResult verifyBindings(mlir::DictionaryAttr bindings,
                                    ConfigurationStage stage) const;
  mlir::LogicalResult verifyRequirements(
      llvm::ArrayRef<ConfigurationRequirementAttr> requirements) const;
  mlir::FailureOr<llvm::SmallVector<ConfigurationRequirementAttr>>
  requirements() const;
  mlir::FailureOr<llvm::SmallVector<mlir::DictionaryAttr>>
  configurations(ConfigurationStage stage) const;

private:
  mlir::func::FuncOp kernel;
  llvm::SmallVector<ParameterAttr> parameters;
  llvm::SmallVector<ParameterAttr> extents;
};

} // namespace intent::gpu
#endif
