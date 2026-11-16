#ifndef INTENT_DIALECT_Block_TRANSFORMS_PROGRAMINTERFACE_H
#define INTENT_DIALECT_Block_TRANSFORMS_PROGRAMINTERFACE_H

#include "Intent/Dialect/Block/IR/ProgramInterface.h"

namespace intent::block {

ArgumentRefAttr nextArgumentReference(mlir::func::FuncOp kernel);
mlir::FailureOr<mlir::BlockArgument> appendArgument(
    mlir::func::FuncOp kernel, mlir::Type type, ArgumentBindingAttr binding);
mlir::LogicalResult setArgumentType(mlir::BlockArgument argument, mlir::Type type);

} // namespace intent::block
#endif
