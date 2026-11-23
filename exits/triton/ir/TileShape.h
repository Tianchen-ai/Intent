#pragma once
#include "mlir/IR/BuiltinTypes.h"
namespace CI_UNIT_NAMESPACE {
inline llvm::ArrayRef<int64_t> shape(mlir::Type t) {
  auto s = mlir::dyn_cast<mlir::RankedTensorType>(t);
  return s ? s.getShape() : llvm::ArrayRef<int64_t>{};
}
} // namespace CI_UNIT_NAMESPACE
