#ifndef INTENT_DIALECT_GPU_IR_FRAGMENTOPINTERFACE_H
#define INTENT_DIALECT_GPU_IR_FRAGMENTOPINTERFACE_H

#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeRange.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace intent::gpu {

// Coordinate/extent relations only: these do not imply equal element values,
// permission to replay a producer, or memory aliasing.
enum class FragmentAxisRelationKind { Corresponding, Broadcast, Reassociation };

struct FragmentAxisGroup {
  FragmentAxisRelationKind kind;
  llvm::SmallVector<unsigned> sourceAxes;
  llvm::SmallVector<unsigned> resultAxes;
};

// A short-lived snapshot taken before changing an operation's operands/types.
// Operand slots remain distinct even when several operands hold the same SSA.
// Reassociation groups include execution-prefix axes in physical coordinates.
struct FragmentOperandRelation {
  unsigned operandNumber;
  mlir::Type sourceType;
  mlir::Type resultType;
  llvm::SmallVector<FragmentAxisGroup> groups;
  // Unary/cast operations forward the sole operand's complete lane schema.
  // Projection operations instead retain their declared result occurrences.
  bool preservesSourceSchema = false;
  // Extents owned by the operation, independent of any operand (e.g. Join's
  // trailing pair). Omitted groups in a partial query do not imply invariance.
  llvm::SmallVector<unsigned, 1> invariantResultAxes = {};

  const FragmentAxisGroup *groupForResultAxis(unsigned axis) const;
  std::optional<unsigned> correspondingSourceAxis(unsigned resultAxis) const;
  bool isIntroducedUnitAxis(unsigned resultAxis) const;
  bool isUnitAxisInsertion() const;
  bool preservesNonUnitAxes() const;
  bool hasCompatibleExtents() const;
};

bool haveEqualPhysicalElementCounts(llvm::ArrayRef<mlir::Attribute> lhs,
                                   llvm::ArrayRef<mlir::Attribute> rhs);

mlir::FailureOr<llvm::SmallVector<FragmentOperandRelation>>
queryFragmentOperandRelations(mlir::Operation *operation,
                              unsigned resultNumber = 0);
mlir::FailureOr<llvm::SmallVector<FragmentOperandRelation>>
queryFragmentOperandRelations(mlir::OpResult result);
mlir::FailureOr<llvm::SmallVector<FragmentOperandRelation>>
queryFragmentOperandRelations(mlir::Operation *operation,
                              mlir::TypeRange operandTypes,
                              mlir::Type resultType,
                              unsigned resultNumber = 0);

// Transport extents through the recorded relation. The destination's declared
// element type is preserved. Projection operations also preserve destination
// logical occurrences, validity and owner; sole-source schema forwarding is
// recorded explicitly above. General split groups are not inverted by guessing
// an axis or dividing symbolic sizes.
// Scalar-to-fragment ownership creation remains the transformation's decision.
// A caller may prove that an old physical singleton and its projected result
// are both being refined within the same selected coordinate domain. This is
// not implied by equal new extents. Slots without a source axis remain broadcast.
using FragmentBroadcastRefinement =
    llvm::function_ref<bool(unsigned operandNumber, unsigned sourceAxis,
                           unsigned resultAxis)>;
mlir::FailureOr<mlir::Type> transportFragmentResultType(
    llvm::ArrayRef<FragmentOperandRelation> relations,
    mlir::TypeRange operandTypes, mlir::Type declaredResult,
    FragmentBroadcastRefinement refineBroadcast = {});
mlir::FailureOr<mlir::Type> transportFragmentOperandType(
    const FragmentOperandRelation &relation, mlir::Type newResultType,
    mlir::Type declaredOperand);

} // namespace intent::gpu

#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h.inc"

#endif
