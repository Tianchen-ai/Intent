#ifndef INTENT_DIALECT_DSA_ANALYSIS_PHYSICALPROGRAM_H
#define INTENT_DIALECT_DSA_ANALYSIS_PHYSICALPROGRAM_H

#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <optional>

namespace intent::dsa {

// Queries read the current program. Recompute after mutations; no target layout
// or allocation binding is written while asking about aliases or lifetimes.
using SignedInterval = std::optional<std::pair<int64_t, int64_t>>;
SignedInterval integerInterval(mlir::Value value, mlir::func::FuncOp function);
// Prove an index/i64 address equals a sum of products after integer algebraic
// normalization. Other operations remain SSA atoms; narrow casts are not peeled.
bool isSumOfIntegerProducts(mlir::Value value,
    llvm::ArrayRef<std::pair<mlir::Value, mlir::Value>> products);
} // namespace intent::dsa
#endif
