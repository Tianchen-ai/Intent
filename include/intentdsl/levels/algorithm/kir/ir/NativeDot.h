#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intentdsl::kir {
// Query the resolved contract of a structurally proven BF16 x BF16 -> FP32 dot.
// A local denormal override takes precedence over the target default.
inline bool permitsBf16Dot(mlir::Operation *op) {
  auto fn = op->getParentOfType<mlir::func::FuncOp>();
  if (!fn)
    return false;
  auto product = fn->getAttrOfType<mlir::StringAttr>("kir.bf16_dot");
  if (product == "fp32-product")
    return false;
  if (op->hasAttr("denormal-fp-math") || product != "native")
    return compilerinfra::allowsSubnormalFlush(op);
  return true;
}
} // namespace intentdsl::kir
