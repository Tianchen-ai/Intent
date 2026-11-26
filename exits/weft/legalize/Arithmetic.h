#pragma once
#include "compilerinfra/Contract.h"
#include "mlir/IR/TypeUtilities.h"

namespace CI_UNIT_NAMESPACE {
// Weft 1.x fuses adjacent multiply/add operations without a per-op switch.
// An explicit cast rounds the exact product before any consuming addition.
template <class Builder, class Term>
Term lowerBinary(Builder &builder, mlir::Operation &op, llvm::StringRef kind,
                 Term lhs, Term rhs, llvm::StringRef resultType) {
  auto element = mlir::getElementTypeOrSelf(op.getResult(0).getType());
  if (kind != "mul" || (!element.isF16() && !element.isBF16() && !element.isF32()))
    return builder.bin(kind, lhs, rhs, resultType);
  bool separate = llvm::any_of(op.getUsers(), [&](mlir::Operation *user) {
    auto n = user->getName().stripDialect();
    return (n == "add" || n == "sub" || n == "addf" || n == "subf") &&
           (!compilerinfra::hasFastMath(&op, mlir::arith::FastMathFlags::contract) ||
            !compilerinfra::hasFastMath(user, mlir::arith::FastMathFlags::contract));
  });
  if (!separate)
    return builder.bin(kind, lhs, rhs, resultType);
  llvm::StringRef wider = element.isF16() ? "f32" : "f64";
  auto wideType = [&](llvm::StringRef type) -> std::string {
    if (!type.starts_with("!weft_kernel.value<"))
      return wider.str();
    return ("!weft_kernel.value<" + wider + "," + type.split(',').second).str();
  };
  auto a = builder.castOp(lhs, wideType(lhs.type));
  auto b = builder.castOp(rhs, wideType(rhs.type));
  return builder.castOp(builder.bin(kind, a, b, wideType(resultType)), resultType);
}
} // namespace CI_UNIT_NAMESPACE
