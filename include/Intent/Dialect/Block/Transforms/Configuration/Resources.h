#ifndef INTENT_DIALECT_Block_TRANSFORMS_RESOURCES_H
#define INTENT_DIALECT_Block_TRANSFORMS_RESOURCES_H

#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::block {

mlir::FailureOr<llvm::SmallVector<mlir::DictionaryAttr>> filterConfigurationRequirements(
    mlir::func::FuncOp kernel, llvm::ArrayRef<mlir::DictionaryAttr> rows,
    llvm::ArrayRef<ConfigurationRequirementAttr> requirements);

} // namespace intent::block

#endif
