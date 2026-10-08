#ifndef INTENT_DIALECT_GPU_TRANSFORMS_EXECUTIONSCHEMA_H
#define INTENT_DIALECT_GPU_TRANSFORMS_EXECUTIONSCHEMA_H

#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeRange.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {

struct LiftedFragmentSchema {
  FragmentType type;
  llvm::SmallVector<unsigned> oldToNew;
  llvm::SmallVector<unsigned> executionToNew;
};

// A short-lived selection of independent execution axes, never a new policy or
// an execution plan. Original axis occurrences remain distinct and ordered.
class ExecutionSchema {
public:
  explicit ExecutionSchema(FragmentType selectedAxes) : selectedAxes(selectedAxes) {}
  mlir::FailureOr<mlir::Type> lift(mlir::Type original) const;
  mlir::FailureOr<LiftedFragmentSchema> project(FragmentType original) const;

private:
  FragmentType selectedAxes;
};

// Infer only the elementwise/product schema established by mapped operands.
// Operations whose result axes are a decision require explicit selected types.
mlir::FailureOr<llvm::SmallVector<mlir::Type>>
inferElementwiseSchema(mlir::Operation *source, mlir::ValueRange mappedOperands);

// Rebind axis-bearing attributes against original types saved before mutation.
// Equal-rank transport retains axis positions; rank expansion embeds existing
// occurrences. This is not a query for an arbitrary axis permutation.
mlir::LogicalResult remapSchemaAxes(mlir::Operation *operation,
                                  mlir::TypeRange originalOperands,
                                  mlir::TypeRange originalResults);

using ClonedAxisRefinement =
    llvm::function_ref<bool(mlir::OpOperand &source, unsigned sourceAxis,
                           mlir::OpResult result, unsigned resultAxis)>;

// Rebuild a leaf with native attributes/properties and real operand projection.
// Explicit selectedResults are authoritative; an empty list requests existing
// elementwise/product inference. Results may be existing SSA values instead of
// results of a new operation. Failure publishes neither new IR nor mappings.
mlir::FailureOr<llvm::SmallVector<mlir::Value>> cloneWithSchema(
    mlir::OpBuilder &builder, mlir::Operation *source, mlir::IRMapping &mapping,
    mlir::TypeRange selectedResults = {});

// Explicit operand occurrences, for example the same source SSA used in two
// contraction roles with different projections. mapping publishes results;
// the operand list is not reconstructed through its Value-keyed bindings.
mlir::FailureOr<llvm::SmallVector<mlir::Value>> cloneWithSchema(
    mlir::OpBuilder &builder, mlir::Operation *source,
    mlir::ValueRange mappedOperands, mlir::IRMapping &mapping,
    mlir::TypeRange selectedResults = {});

// Rebuild an operation and its regions at the current insertion scope. select
// supplies the declared base type of each original result and block argument
// (null means unsupported); unchanged source relations and actual operands then
// refine results. Unlike explicit leaf selectedResults, an unbound extent may
// therefore follow its actual producer. The source stays the relation snapshot.
// Region/control organization and replay legality remain the caller's choice.
// Leaves share cloneWithSchema's reconstruction, while native region cloning
// preserves the original control/terminator slots. Callers still finish range
// bounds, active masks and other execution decisions before group verification.
// Only successful completion
// updates mapping, including nested results/arguments and reused values.
mlir::FailureOr<llvm::SmallVector<mlir::Value>> cloneWithPhysicalSchema(
    mlir::OpBuilder &builder, mlir::Operation *source, mlir::IRMapping &mapping,
    llvm::function_ref<mlir::Type(mlir::Value)> select,
    ClonedAxisRefinement refineBroadcast = {});

} // namespace intent::gpu
#endif
