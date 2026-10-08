#ifndef INTENT_DIALECT_GPU_TRANSFORMS_VALUERELATIONS_H
#define INTENT_DIALECT_GPU_TRANSFORMS_VALUERELATIONS_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/Value/SchemaMutation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {

// Intermediate rewrites sometimes need only the relations they have made
// authoritative. Complete transformation entries close every relation.
enum class ValueRelationScope {
  Complete, Pointwise, Contracts, AccessResults, ReductionInputs
};

// Relation maintenance belongs inside a complete transformation, before verify.
// The selected occurrence is positional. Repeated logical/source identities in
// another axis do not join their independent physical extents.
mlir::LogicalResult retargetFragmentAxisExtent(
    mlir::Value root, unsigned fragmentAxis, PhysicalExprAttr extent,
    ValueTypeChangeCallback changed = {},
    mlir::OpBuilder::Listener *listener = nullptr);
// Preserve the value's scalar/record structure and coordinate occurrences,
// while applying each selected leaf extent independently.
mlir::LogicalResult retargetValueExtents(
    mlir::Value root, mlir::Type selected,
    ValueTypeChangeCallback changed = {},
    mlir::OpBuilder::Listener *listener = nullptr);
mlir::LogicalResult retargetSourceExtent(mlir::Value root, PhysicalSourceAxis source,
                          PhysicalExprAttr extent,
                          std::optional<int64_t> dimension = std::nullopt,
                          ValueTypeChangeCallback changed = {},
                          mlir::OpBuilder::Listener *listener = nullptr);
mlir::LogicalResult retargetDimensionExtent(mlir::Value root, int64_t dimensionId,
                             PhysicalExprAttr extent,
                             ValueTypeChangeCallback changed = {},
                             mlir::OpBuilder::Listener *listener = nullptr);
mlir::LogicalResult closeValueRelations(
    mlir::func::FuncOp kernel,
    ValueRelationScope scope = ValueRelationScope::Complete);
void eraseDeadPhysicalValues(mlir::func::FuncOp kernel);

} // namespace intent::gpu

#endif
