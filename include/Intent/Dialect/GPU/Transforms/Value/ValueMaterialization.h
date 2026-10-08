#ifndef INTENT_DIALECT_GPU_TRANSFORMS_VALUEMATERIALIZATION_H
#define INTENT_DIALECT_GPU_TRANSFORMS_VALUEMATERIALIZATION_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/IR/IRMapping.h"

namespace intent::gpu {

// Rewrites consume current coordinate/replay facts; they do not choose tiling.
struct ReplayMaterializationOptions {
  PhysicalReplayScope scope = PhysicalReplayScope::ValueGraph;
  bool allowAccesses = true;
  llvm::ArrayRef<MakeRangeOp> traversalRanges;
  std::optional<unsigned> fragmentAxis;
  mlir::Value segmentTail;
  AxisMapAttr segmentMapping;
  bool materializeZeroFill = false;
};

void foldExactConstantDivisions(mlir::func::FuncOp kernel);
void foldScalarIntegerValues(mlir::func::FuncOp kernel);
mlir::FailureOr<mlir::Value>
materializeNonOverlappingView(mlir::func::FuncOp kernel, mlir::Value resource);
mlir::FailureOr<mlir::Value>
materializeScalarConstant(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Attribute value, mlir::Type resultType);
// Expands one coordinate occurrence into the access's payload lane domain.
// Keeps scalar coordinates scalar and never replays or retile its producer.
mlir::FailureOr<mlir::Value>
materializeAccessCoordinate(mlir::OpBuilder &builder, AccessOpInterface access,
                            unsigned coordinateIndex);
mlir::FailureOr<mlir::Value>
projectPhysicalValueToSchema(mlir::OpBuilder &builder, mlir::Location location,
                             mlir::Value value, mlir::Type target,
                             ValueTypeChangeCallback changed = {});
mlir::FailureOr<mlir::Value>
materializeZeroValue(mlir::OpBuilder &builder, mlir::Location location,
                     mlir::Type target);
mlir::FailureOr<mlir::Value>
materializeZeroFragment(mlir::OpBuilder &builder, mlir::Location location,
                        FragmentType target);
// insertionAnchor is the original semantic position of the rewrite. The
// builder may be in a newly constructed block, including at its empty end.
// Prebound mappings are preserved snapshots, available at that actual point.
mlir::FailureOr<mlir::Value> materializeReplayedValue(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    PhysicalSourceAxis source, PhysicalExprAttr blockedExtent,
    mlir::IRMapping &mapping, mlir::Operation *insertionAnchor,
    ReplayMaterializationOptions options = {});
// Replay all occurrences selected by one exact set of range roots. This keeps
// the range-set projection distinct from the single occurrence API above.
mlir::FailureOr<mlir::Value> materializeReplayedRanges(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    PhysicalExprAttr blockedExtent, llvm::ArrayRef<MakeRangeOp> roots,
    mlir::Value replacement, mlir::IRMapping &mapping,
    mlir::Operation *insertionAnchor);
mlir::FailureOr<mlir::Value>
projectPredicateToFragmentAxis(mlir::OpBuilder &builder, mlir::Location location,
                               mlir::Value predicate, FragmentType target,
                               unsigned fragmentAxis);
mlir::FailureOr<mlir::Value>
projectPredicateToFragment(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value predicate, FragmentType target,
                           PhysicalSourceAxis source);
mlir::FailureOr<mlir::Value>
projectPredicateToFragment(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value predicate, FragmentType target,
                           int64_t dimensionId);
mlir::FailureOr<mlir::Value>
materializeValidityConjunction(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value lhs,
                               mlir::Value rhs, FragmentType valueType);
// Replaces proven tail predicates while retaining author scalar predicates.
mlir::FailureOr<mlir::Value> materializeRetargetedValidity(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value original,
    llvm::ArrayRef<std::pair<MakeRangeOp, mlir::Value>> originalTailRanges,
    mlir::Value physicalTail, FragmentType target);

} // namespace intent::gpu

#endif
