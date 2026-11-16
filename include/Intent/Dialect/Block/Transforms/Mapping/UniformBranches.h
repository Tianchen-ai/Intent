#ifndef INTENT_Block_TRANSFORMS_MAPPING_UNIFORMBRANCHES_H
#define INTENT_Block_TRANSFORMS_MAPPING_UNIFORMBRANCHES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::block {

mlir::scf::IfOp independentUniformBranches(mlir::func::FuncOp kernel);
// Reuse the caller's shared transformations for independent branch programs,
// then rejoin their mapping under the original single launch and public ABI.
mlir::FailureOr<bool> realizeUniformBranches(mlir::ModuleOp module,
    mlir::func::FuncOp kernel, mlir::scf::IfOp conditional,
    llvm::function_ref<mlir::LogicalResult(mlir::ModuleOp)> runTransforms);

} // namespace intent::block
#endif
