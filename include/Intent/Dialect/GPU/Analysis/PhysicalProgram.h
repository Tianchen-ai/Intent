#ifndef INTENT_DIALECT_GPU_ANALYSIS_PHYSICALPROGRAM_H
#define INTENT_DIALECT_GPU_ANALYSIS_PHYSICALPROGRAM_H

#include "Intent/Analysis/ContractionAxes.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>

namespace intent::gpu {

mlir::FailureOr<mlir::func::FuncOp> getPhysicalKernel(mlir::ModuleOp module);

// Read the common axis schema from the current ordinary/scaled/sparse op.
// This adapter checks ranks only; format-specific extent rules stay in the IR.
std::optional<ContractionAxes>
queryContractionAxes(mlir::Operation *operation,
                     std::string *failureReason = nullptr);

/// A logical source axis carried by the current executable GPU program.
struct PhysicalSourceAxis {
  uint64_t sourceId = 0;
  uint64_t sourceAxis = 0;
  bool derived = false;

  bool operator==(const PhysicalSourceAxis &other) const {
    return sourceId == other.sourceId && sourceAxis == other.sourceAxis &&
           derived == other.derived;
  }
};

PhysicalSourceAxis sourceAxisIdentity(AxisMapAttr mapping);
PhysicalSourceAxis sourceAxisIdentity(MakeRangeOp range);

enum class PhysicalFactState { Exact, Unknown, Ambiguous };

enum class PhysicalReplayScope {
  /// Scalar/coordinate expressions and access coordinates.  Structured
  /// reductions and contractions are not mechanically replayable here.
  Coordinate,
  /// A complete pointwise value graph, including first-class structured
  /// operations whose result type can be retargeted by the caller.
  ValueGraph,
};

bool isPhysicalReplayNode(mlir::Operation *operation,
                          PhysicalReplayScope scope,
                          bool allowAccesses);

/// Includes writes in structured iterations entered between the read and use.
/// ABI aliases are conservatively treated as potentially overlapping.
/// Moving to an earlier anchor also requires the caller to make operands
/// available there; this query only establishes memory stability.
bool canReplayReadAt(LoadOp load, mlir::Operation *insertionAnchor);

/// Proves non-aliasing for private-buffer accesses, including a uniform
/// coordinate excluded by the other access's current typed validity.
/// The caller must preserve the evaluated SSA coordinates while moving effects.
bool haveDisjointPrivateBufferAccesses(mlir::Operation *lhs,
                                       mlir::Operation *rhs);

struct PhysicalAxisProjection {
  PhysicalFactState state = PhysicalFactState::Unknown;
  PhysicalSourceAxis source;
  int64_t dimensionId = 0;
  unsigned fragmentAxis = 0;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

struct PhysicalDimensionProjection {
  PhysicalFactState state = PhysicalFactState::Unknown;
  int64_t dimensionId = 0;
  unsigned fragmentAxis = 0;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

PhysicalAxisProjection queryFragmentAxis(mlir::Type type,
                                         PhysicalSourceAxis source,
                                         std::optional<int64_t> dimension = std::nullopt);
llvm::SmallVector<PhysicalAxisProjection, 2>
queryFragmentAxes(mlir::Type type, PhysicalSourceAxis source);
llvm::SmallVector<PhysicalAxisProjection, 2>
queryRangeProjections(mlir::Type type, MakeRangeOp range);
PhysicalDimensionProjection queryFragmentDimension(mlir::Type type,
                                                   int64_t dimensionId);
llvm::SmallVector<PhysicalDimensionProjection, 2>
queryFragmentDimensions(mlir::Type type, int64_t dimensionId);
mlir::FailureOr<int64_t>
querySourceDimension(mlir::Type type, PhysicalSourceAxis source);
PhysicalAxisProjection
queryCoordinateIndex(mlir::ValueRange coordinates, PhysicalSourceAxis source,
                     std::optional<int64_t> dimension = std::nullopt);
mlir::FailureOr<unsigned>
queryCoordinatePosition(mlir::ValueRange coordinates,
                        PhysicalSourceAxis source);
mlir::FailureOr<AxisMapAttr> queryAxisMap(mlir::Type type,
                                         unsigned fragmentAxis);
mlir::FailureOr<int64_t> queryRangeDimension(MakeRangeOp range);
bool samePhysicalScalarExpression(mlir::Value lhs, mlir::Value rhs);
bool isKnownPositiveExtent(PhysicalExprAttr extent, mlir::func::FuncOp kernel);
std::optional<std::pair<int64_t, int64_t>>
queryPositiveExtentBounds(PhysicalExprAttr extent, mlir::func::FuncOp kernel);
bool isLaunchUniformScalar(mlir::Value value, mlir::func::FuncOp kernel);

/// Prove one launch instance and a singleton execution-group coordinate domain.
/// This does not establish legality for moving effects across surrounding control.
bool isSingletonExecutionGroup(ExecutionGroupOp group, mlir::func::FuncOp kernel);

/// Exact launch-visible integer expression from current scalar SSA. Device
/// accesses, program coordinates and ordered carries are not launch leaves.
PhysicalExprAttr queryLaunchExpression(mlir::Value value);

/// Exact logical member count when the range bounds are launch-visible.
PhysicalExprAttr queryLaunchRangeExtent(MakeRangeOp range);

/// A compile-time inclusive upper bound for a proven non-negative scalar
/// index, derived from current mapping, dimensions and clamps. Null means unknown.
PhysicalExprAttr queryNonNegativeIndexUpperBound(mlir::Value value);

/// Launch-visible capacity for a unit-step logical range. Logical bounds and
/// validity remain unchanged; the host evaluates this bound before allocation.
PhysicalExprAttr queryLogicalRangeCapacity(MakeRangeOp range);

/// All current-IR range roots that carry one requested source axis.
struct PhysicalRangeFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  llvm::SmallVector<MakeRangeOp, 2> roots;
  llvm::SmallVector<mlir::Operation *, 2> accesses;
  llvm::SmallVector<mlir::Operation *, 2> blockers;
  bool unitStep = false;

  bool isExact() const { return state == PhysicalFactState::Exact; }
  bool isUnique() const { return isExact() && roots.size() == 1; }
};

/// Whether one current fragment axis has been materialized over its exact
/// physical range.  Construction may legally seed a dynamic logical axis with
/// a scalar fragment; that scalar type is not evidence that the traversal has
/// already been blocked.  Consumers use this fact instead of comparing shape
/// attributes or range operands independently.
struct PhysicalAxisRealizationFact {
  enum class ExtentAuthority {
    None,
    /// The current axis is materialized by one exact physical range whose SSA
    /// extent agrees with the fragment extent.
    Range,
    /// A verified value relation explicitly selects or preserves this physical
    /// extent.  This includes reshape reassociation and an extent-preserving
    /// projection of an already authoritative input.
    Structural,
  };

  PhysicalFactState state = PhysicalFactState::Unknown;
  PhysicalSourceAxis source;
  int64_t dimensionId = 0;
  unsigned fragmentAxis = 0;
  llvm::SmallVector<MakeRangeOp, 2> roots;
  llvm::SmallVector<mlir::Operation *, 2> blockers;
  bool constructionScalarSeed = false;
  bool physicalized = false;
  ExtentAuthority extentAuthority = ExtentAuthority::None;

  bool isExact() const { return state == PhysicalFactState::Exact; }
  bool hasExtentAuthority() const {
    return isExact() && extentAuthority != ExtentAuthority::None;
  }
};

/// Exact fragment axes whose current coordinate provenance reaches one of a
/// selected set of physical range roots.  This distinguishes repeated uses of
/// one logical source axis in different result positions without inventing a
/// second source identity.
struct PhysicalRangeAxisFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  llvm::SmallVector<unsigned, 2> fragmentAxes;
  llvm::SmallVector<mlir::Operation *, 2> blockers;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

enum class PhysicalLockstepState { Exact, Unknown, Inconsistent };

/// One current physical traversal shared by several structured sources.  The
/// selected MakeRange operation is only a canonical SSA carrier after all
/// source ranges have been proven equivalent.
struct PhysicalLockstepTraversalFact {
  PhysicalLockstepState state = PhysicalLockstepState::Unknown;
  MakeRangeOp authority;
  llvm::SmallVector<mlir::Operation *, 4> blockers;

  bool isExact() const { return state == PhysicalLockstepState::Exact; }
};

bool sameLogicalRange(MakeRangeOp lhs, MakeRangeOp rhs);
bool isUnitStepRange(MakeRangeOp range);
std::optional<int64_t> constantLogicalRangeCardinality(MakeRangeOp range);
bool isProvablySingletonLogicalRange(MakeRangeOp range);
/// The launch-visible parent dimension of a narrowed logical subregion.  The
/// subregion keeps its own extent identity in RangeType/AxisMapAttr; this
/// relation only supplies the program-space upper bound used for ownership.
mlir::FailureOr<int64_t> querySubregionParentDimension(MakeRangeOp range);
mlir::FailureOr<MakeRangeOp> queryExactLogicalRange(
    const PhysicalRangeFact &fact);

/// Whether a value can be mechanically rebuilt after replacing one source
/// range.  This is a fact about the current graph; it never performs cloning.
struct PhysicalReplayFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  bool crossesAccess = false;
  bool crossesStructuredProgram = false;
  llvm::SmallVector<mlir::Operation *, 2> accesses;
  llvm::SmallVector<mlir::Operation *, 2> contractions;
  llvm::SmallVector<mlir::Operation *, 2> structuredPrograms;
  llvm::SmallVector<mlir::Operation *, 2> blockers;

  bool isReplayable() const { return state == PhysicalFactState::Exact; }
};

/// Exact dependence of one current physical value on a structured reduction
/// traversal. Unknown is conservative and carries the operations that prevent
/// the relation from being established.
struct PhysicalReductionDependencyFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  bool depends = false;
  bool throughStructuredReduction = false;
  llvm::SmallVector<mlir::Operation *, 2> blockers;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

/// Current physical realizations of every non-batch, non-reduction operand
/// axis of one contraction operation.  This is the single authority used by
/// ownership and contraction transformations; neither consumer re-derives
/// free axes from result shape or nearby stores.
struct PhysicalContractFreeAxis {
  mlir::Value operand;
  unsigned operandAxis = 0;
  PhysicalAxisRealizationFact realization;
  PhysicalRangeFact ranges;
};

struct PhysicalContractFreeAxisFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  llvm::SmallVector<PhysicalContractFreeAxis, 4> axes;
  llvm::SmallVector<mlir::Operation *, 2> blockers;

  bool isExact() const { return state == PhysicalFactState::Exact; }
  bool needsRealization() const {
    return !isExact() || llvm::any_of(axes, [](const auto &axis) {
             return !axis.realization.physicalized;
           });
  }
};

/// Typed logical relation carried by one physical parameter declaration.  The
/// parameter name remains only its compile-time symbol; consumers must not
/// recover a dimension or source axis from that spelling.
struct PhysicalParameterBinding {
  PhysicalFactState state = PhysicalFactState::Unknown;
  std::optional<int64_t> dimension;
  std::optional<PhysicalSourceAxis> source;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

PhysicalParameterBinding queryParameterBinding(ParameterAttr parameter);
ParameterAttr queryParameter(mlir::Value value);
mlir::FailureOr<ParameterAttr>
queryParameterBySymbol(mlir::func::FuncOp kernel, mlir::StringAttr symbol);
mlir::FailureOr<ParameterAttr>
queryBlockingParameter(mlir::func::FuncOp kernel, MakeRangeOp range);

/// Exact current-IR access relation, or an explicit conservative result.
struct PhysicalAccessFootprint {
  PhysicalFactState state = PhysicalFactState::Unknown;
  PhysicalFactState rangeState = PhysicalFactState::Unknown;
  mlir::Value resource;
  llvm::SmallVector<mlir::Value, 4> coordinates;
  llvm::SmallVector<int64_t, 4> sourceAxes;
  llvm::SmallVector<MakeRangeOp, 4> ranges;
  llvm::SmallVector<mlir::Operation *, 4> blockers;
  mlir::Value validity;
  mlir::Value fill;
};

/// Whether one access validity is either unconditional or is composed only of
/// exact current-coordinate upper bounds that a provider-native boundary form
/// can preserve.  `boundaryAxes` names the external-view axes whose bounds
/// must remain active; exact physical-range bounds that are already true for
/// every represented member do not add an axis.
struct PhysicalAccessBoundaryFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  llvm::SmallVector<int64_t, 4> boundaryAxes;
  /// Requested conditional bounds: each entire unit-step range must lie below
  /// its exclusive upper bound before replacing the original member predicate.
  llvm::SmallVector<std::pair<MakeRangeOp, mlir::Value>, 2> rangeBounds;
  llvm::SmallVector<mlir::Operation *, 4> blockers;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

/// Resource-bounds legality for one current physical access.  Exact means
/// every resource axis is covered by an access predicate, an exact current
/// coordinate range, or a dominating typed in-bounds precondition.  Unlike
/// PhysicalAccessBoundaryFact, unrelated logical predicates do not make this
/// fact unknown: they can only further restrict the active member set.
struct PhysicalAccessBoundsFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  /// Resource axes closed by a dominating typed AssumeInBoundsOp.
  llvm::SmallVector<int64_t, 4> assumedAxes;
  llvm::SmallVector<int64_t, 4> unprovenAxes;
  llvm::SmallVector<int64_t, 4> missingLowerAxes;
  llvm::SmallVector<int64_t, 4> missingUpperAxes;
  llvm::SmallVector<mlir::Operation *, 4> blockers;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

/// Recomputable facts derived only from the current executable GPU IR.
/// A transformation constructs this object after the preceding mutation,
/// queries all plans it needs, then discards it before rewriting the IR.
class PhysicalProgramAnalysis {
public:
  explicit PhysicalProgramAnalysis(mlir::func::FuncOp kernel);

  mlir::FailureOr<unsigned> fragmentAxis(mlir::Type type,
                                         PhysicalSourceAxis source) const;
  mlir::FailureOr<unsigned>
  coordinateIndex(mlir::ValueRange coordinates,
                  PhysicalSourceAxis source) const;
  mlir::FailureOr<unsigned> accessCoordinatePosition(
      LoadOp load, AxisMapAttr mapping, mlir::Value operand);
  PhysicalRangeFact sourceRanges(
      mlir::Value value,
      std::optional<PhysicalSourceAxis> source = std::nullopt);
  /// Exact current-program ranges for a typed logical source axis.  This is
  /// used only when a replayable pure value carries the axis in its type but
  /// has no producer edge to the coordinate value.
  PhysicalRangeFact programRanges(PhysicalSourceAxis source);
  /// Exact range roots that make one concrete fragment axis vary.  Unlike a
  /// source-id query, this preserves repeated occurrences of the same logical
  /// source in Cartesian/indexed values.
  PhysicalRangeFact axisRanges(mlir::Value value, unsigned fragmentAxis);
  PhysicalAxisRealizationFact axisRealization(mlir::Value value,
                                               unsigned fragmentAxis);
  PhysicalRangeAxisFact rangeAxes(mlir::Value value,
                                  llvm::ArrayRef<MakeRangeOp> roots);
  PhysicalLockstepTraversalFact
  lockstepRanges(llvm::ArrayRef<MakeRangeOp> ranges);
  PhysicalLockstepTraversalFact
  lockstepTraversal(mlir::ValueRange sources,
                    llvm::ArrayRef<unsigned> fragmentAxes);
  /// Query the source graph independently of a destination. This establishes
  /// structural eligibility only; an actual rewrite must use replayAt.
  PhysicalReplayFact replayability(
      mlir::Value value,
      std::optional<PhysicalSourceAxis> source = std::nullopt,
      PhysicalReplayScope scope = PhysicalReplayScope::Coordinate,
      bool allowAccesses = true,
      std::optional<int64_t> sourceDimension = std::nullopt);
  /// Prove motion to an existing semantic position. Bound SSA values are the
  /// completed frontier of this rewrite and are reused, not read again.
  /// Unlike a structural replayability query, this requires a non-null anchor.
  /// With no source selector, every unbound shaped producer is replayed;
  /// otherwise independent values that dominate the anchor may be reused.
  PhysicalReplayFact replayAt(
      mlir::Value value, std::optional<PhysicalSourceAxis> source,
      PhysicalReplayScope scope, bool allowAccesses,
      mlir::Operation *insertionAnchor, const mlir::IRMapping &bindings,
      std::optional<int64_t> sourceDimension = std::nullopt);
  PhysicalReductionDependencyFact reductionDependency(
      mlir::Value value, PhysicalSourceAxis source,
      std::optional<int64_t> sourceDimension = std::nullopt);
  PhysicalContractFreeAxisFact contractFreeAxes(mlir::Operation *contract);
  PhysicalAccessFootprint footprint(mlir::Operation *access);
  /// Conditional range bounds are opt-in: the caller must materialize their
  /// guards before using the boundary form, retaining the original access otherwise.
  PhysicalAccessBoundaryFact boundaryValidity(mlir::Operation *access,
                                             bool allowRangeGuards = false);
  PhysicalAccessBoundsFact accessBounds(mlir::Operation *access);
  bool isProgramOwnedRange(MakeRangeOp range) const;
  bool hasDisjointWorkspaceSlices(mlir::Value buffer) const;

  /// Recognizes a predicate composed only from exact range-bound comparisons,
  /// predicate-preserving shape operations and boolean conjunction.  This is
  /// a pure query; it never invents or rewrites validity.
  bool isTailPredicate(
      mlir::Value value,
      llvm::ArrayRef<std::pair<MakeRangeOp, mlir::Value>> ranges) const;

private:
  using ReplayContext = std::pair<std::optional<PhysicalSourceAxis>,
                                  std::optional<int64_t>>;
  using ReplayVisits = llvm::DenseMap<mlir::Operation *,
                                      llvm::SmallVector<ReplayContext, 2>>;

  void collectRanges(mlir::Value value,
                     std::optional<PhysicalSourceAxis> source,
                     PhysicalRangeFact &result,
                     llvm::SmallPtrSetImpl<mlir::Operation *> &visited,
                     bool followScalarDependencies = true);
  void collectAxisRanges(mlir::Value value, unsigned fragmentAxis,
                         PhysicalRangeFact &result,
                         llvm::DenseSet<std::pair<mlir::Value, unsigned>> &visited);
  void analyzeReplay(mlir::Value value,
                     std::optional<PhysicalSourceAxis> source,
                     PhysicalReplayScope scope, bool allowAccesses,
                     mlir::Operation *insertionAnchor,
                     std::optional<int64_t> sourceDimension,
                     mlir::DominanceInfo *dominance,
                     const mlir::IRMapping *bindings,
                     PhysicalReplayFact &result,
                     ReplayVisits &visited);
  llvm::SmallVector<mlir::Value, 2>
  structuredSourcesForArgument(mlir::BlockArgument argument) const;
  bool carriesSource(mlir::Type type, PhysicalSourceAxis source) const;

  mlir::func::FuncOp kernel;
  llvm::DenseMap<mlir::Value, PhysicalRangeFact> unrestrictedRangeCache;
};

} // namespace intent::gpu

namespace llvm {
template <> struct DenseMapInfo<intent::gpu::PhysicalSourceAxis> {
  static inline intent::gpu::PhysicalSourceAxis getEmptyKey() {
    return {DenseMapInfo<uint64_t>::getEmptyKey(),
            DenseMapInfo<uint64_t>::getEmptyKey(), false};
  }
  static inline intent::gpu::PhysicalSourceAxis getTombstoneKey() {
    return {DenseMapInfo<uint64_t>::getTombstoneKey(),
            DenseMapInfo<uint64_t>::getTombstoneKey(), false};
  }
  static unsigned getHashValue(const intent::gpu::PhysicalSourceAxis &value) {
    return static_cast<unsigned>(
        hash_combine(value.sourceId, value.sourceAxis, value.derived));
  }
  static bool isEqual(const intent::gpu::PhysicalSourceAxis &lhs,
                      const intent::gpu::PhysicalSourceAxis &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif
