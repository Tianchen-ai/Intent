#ifndef INTENT_DIALECT_GPU_ANALYSIS_INDEXRELATIONS_H
#define INTENT_DIALECT_GPU_ANALYSIS_INDEXRELATIONS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace intent::gpu {

// Read-only relations over the current typed index graph. Unknown arithmetic,
// invalid divisors and unrepresentable shape expressions do not prove a fact.
// These queries neither insert guards nor choose a provider access form.
// Provider-rebound ResidentWorkers candidates are not a fixed analysis domain;
// only their positive-capacity contract and domain-independent proofs are used.
class IndexRelations {
public:
  std::optional<int64_t> constant(mlir::Value value) const;
  std::optional<int64_t> constant(PhysicalExprAttr expression,
                                  mlir::func::FuncOp kernel) const;
  bool same(mlir::Value lhs, mlir::Value rhs) const;
  bool nonnegative(mlir::Value value) const;
  bool positive(mlir::Value value) const;
  bool atMost(mlir::Value lhs, mlir::Value rhs) const;
  bool lessThan(mlir::Value lhs, mlir::Value rhs) const;
  bool atMost(mlir::Value lhs, PhysicalExprAttr rhs) const;

  // Every represented lane must satisfy the relation. These queries combine
  // current loop/range correlations with fixed-width integer facts; independent
  // intervals are sufficient evidence, never a prerequisite for a relation.
  bool coordinateLessThan(mlir::Value coordinate, mlir::Value limit) const;
  bool coordinateLessThan(mlir::Value coordinate, PhysicalExprAttr limit) const;
  bool coordinateInBounds(mlir::Value coordinate, PhysicalExprAttr extent) const;
  bool hasExactRangeEndpoint(mlir::Value coordinate, mlir::Value limit) const;

  // Return the dividend of a proved nonnegative round-down expression.
  mlir::Value roundedDownSource(mlir::Value value) const;
  bool powerOfTwo(mlir::Value value) const;
  bool multipleOf(mlir::Value value, mlir::Value divisor) const;

  // A positive power-of-two extent and an aligned scalar index origin prove
  // that start + [0, extent - 1] does not cross signed index wraparound. This
  // concerns the represented origin, not the arithmetic that produced it, and
  // proves neither nonnegativity nor resource bounds.
  bool alignedUnitWindow(mlir::Value start, mlir::Value extent) const;

  // An optional leaf proof is valid only within the caller's already guarded
  // specialization. It is consulted for typed parameter declarations, never
  // persisted as a fact about the unrestricted parameter domain.
  bool multipleOf(mlir::Value value, int64_t divisor,
                  llvm::function_ref<bool(ParameterAttr)> alignedParameter = {}) const;

  // Return an equal aligned endpoint, including a selected min arm whose
  // ordering is proven. This does not prove that an entire loop is nonempty.
  mlir::Value alignedBound(mlir::Value value, mlir::Value step) const;

};

} // namespace intent::gpu
#endif
