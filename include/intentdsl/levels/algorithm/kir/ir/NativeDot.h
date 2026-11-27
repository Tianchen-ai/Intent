#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intentdsl::kir {
// Matrix units whose manuals leave subnormal handling unspecified need an
// explicit widened product when the source environment must be preserved.
inline bool requiresSourceDot(mlir::Operation *op, mlir::Type element) {
  auto fn = op->getParentOfType<mlir::func::FuncOp>();
  bool scaled = op->getName().stripDialect() == "scaled_contract";
  if (!fn || (!scaled && !element.isBF16() && !element.isF16()))
    return false;
  auto mode =
      fn->getAttrOfType<mlir::StringAttr>(scaled             ? "kir.fp8_dot"
                                          : element.isBF16() ? "kir.bf16_dot"
                                                             : "kir.f16_dot");
  if (!mode)
    return false;
  return mode == "fp32-product" ||
         (op->hasAttr("denormal-fp-math")
              ? !compilerinfra::allowsSubnormalFlush(op)
              : mode == "source");
}

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
