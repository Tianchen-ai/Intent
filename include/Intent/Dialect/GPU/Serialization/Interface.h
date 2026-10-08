#ifndef INTENT_DIALECT_GPU_SERIALIZATION_INTERFACE_H
#define INTENT_DIALECT_GPU_SERIALIZATION_INTERFACE_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace intent::gpu {

// The expression tree preserves IR symbols and operators. It is data for the
// runtime evaluator, never a generated Python expression or a new policy.
llvm::json::Value serializeExpression(PhysicalExprAttr expression);

// Exports the common public contract and the current GPU argument bindings.
// The provider appends its own named object for native launch details. The
// callback reports the names actually serialized,
// including packed-argument projections when those are part of the native ABI.
mlir::FailureOr<llvm::json::Object> serializeInterface(
    mlir::func::FuncOp kernel, llvm::StringRef provider,
    llvm::function_ref<std::string(mlir::Value)> kernelName);

} // namespace intent::gpu
#endif
