#pragma once

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/JSON.h"
#include "Intent/Serialization/NativeABI.h"

namespace intent::weft_provider {
mlir::FailureOr<llvm::json::Object> serializeTaskABI(::weft::kernel::KernelOp kernel);
mlir::FailureOr<NativeABI> queryHostABI(mlir::func::FuncOp function);
}
