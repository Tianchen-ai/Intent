#ifndef INTENT_DIALECT_CPU_TRANSFORMS_VECTOR_VECTORREDUCTIONS_H
#define INTENT_DIALECT_CPU_TRANSFORMS_VECTOR_VECTORREDUCTIONS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace intent::cpu {

struct IndependentReduction {
  mlir::vector::CombiningKind kind;
  mlir::arith::FastMathFlags fastMath;
};

// Match only a direct same-component binary combine. Reassociation and element
// permutation are separate permissions that the caller must establish.
std::optional<IndependentReduction> matchIndependentReduction(
    mlir::Value lhs, mlir::Value rhs, mlir::Value result);

// Reduce adjacent lanes of a nonempty tuple of fixed rank-one vectors with
// equal power-of-two widths. The callback combines corresponding components;
// source permutation and accumulator initialization remain caller decisions.
// A complete independent list permits unordered native reductions instead of
// the adjacent tree. An empty list keeps the generic, potentially coupled body.
llvm::SmallVector<mlir::Value> horizontalReduce(
    mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange values,
    llvm::function_ref<llvm::SmallVector<mlir::Value>(
        mlir::ValueRange, mlir::ValueRange, int64_t)> combine,
    llvm::ArrayRef<IndependentReduction> independent = {});

} // namespace intent::cpu

#endif
