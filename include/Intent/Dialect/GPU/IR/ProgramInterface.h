#ifndef INTENT_DIALECT_GPU_IR_PROGRAMINTERFACE_H
#define INTENT_DIALECT_GPU_IR_PROGRAMINTERFACE_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

ArgumentBindingAttr getArgumentBinding(mlir::BlockArgument argument);
ArgumentRefAttr getArgumentReference(mlir::Value value);
mlir::BlockArgument resolveArgument(mlir::func::FuncOp kernel,
                                   ArgumentRefAttr reference);
mlir::BlockArgument resolveDimension(mlir::func::FuncOp kernel, int64_t identity);
PhysicalExprAttr queryArgumentExpression(mlir::BlockArgument argument);
intent::PublicParameterAttr publicParameter(mlir::Value value);
intent::ViewType getPublicView(mlir::Value value);
bool isInvocationWorkspace(mlir::Value value);
mlir::LogicalResult verifyProgramInterface(mlir::func::FuncOp kernel);

} // namespace intent::gpu
#endif
