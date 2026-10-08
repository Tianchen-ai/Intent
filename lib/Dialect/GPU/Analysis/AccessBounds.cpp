#include "AccessRelations.h"
#include "IndexBounds.h"
#include "PhysicalProgramDetail.h"
#include "ScalarExpressions.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu {
using namespace detail;

bool PhysicalProgramAnalysis::isTailPredicate(
    Value value, ArrayRef<std::pair<MakeRangeOp, Value>> ranges) const {
  if (!value)
    return true;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    return integer && integer.getType().isInteger(1) && integer.getInt() != 0;
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return isTailPredicate(broadcast.getValue(), ranges);
  if (auto splat = value.getDefiningOp<SplatOp>())
    return isTailPredicate(splat.getValue(), ranges);
  if (auto reshape = value.getDefiningOp<ReshapeOp>())
    return isTailPredicate(reshape.getValue(), ranges);
  if (auto transpose = value.getDefiningOp<TransposeOp>())
    return isTailPredicate(transpose.getValue(), ranges);
  if (auto conjunction = value.getDefiningOp<BinaryOp>()) {
    Type element = conjunction.getResult().getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    bool logical =
        conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
    bool bitwiseI1 =
        conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
        element.isInteger(1);
    return (logical || bitwiseI1) &&
           isTailPredicate(conjunction.getLhs(), ranges) &&
           isTailPredicate(conjunction.getRhs(), ranges);
  }
  auto comparison = value.getDefiningOp<CompareOp>();
  if (!comparison ||
      (comparison.getPredicate() != ComparePredicate::Lt &&
       comparison.getPredicate() != ComparePredicate::Ge))
    return false;
  Value lhs = stripBroadcast(comparison.getLhs());
  Value rhs = stripScalarIdentity(comparison.getRhs());
  MakeRangeOp predicateRange = lhs.getDefiningOp<MakeRangeOp>();
  auto sameTailEnd = [&](Value current, Value expected) {
    if (sameScalarExpression(current, expected))
      return true;
    if (auto dim = current.getDefiningOp<DimOp>())
      return matchesResourceExtent(expected, dim.getView(), dim.getAxis());
    if (auto dim = expected.getDefiningOp<DimOp>())
      return matchesResourceExtent(current, dim.getView(), dim.getAxis());
    return false;
  };
  return llvm::any_of(ranges, [&](const auto &entry) {
    MakeRangeOp expectedRange = entry.first;
    Value expectedEnd = stripScalarIdentity(entry.second);
    bool sameRange = lhs == expectedRange.getResult();
    if (!sameRange && predicateRange &&
        sourceAxisIdentity(predicateRange) ==
            sourceAxisIdentity(expectedRange))
      sameRange = sameScalarExpression(predicateRange.getStart(),
                                       expectedRange.getStart()) &&
                  sameScalarExpression(predicateRange.getExtent(),
                                       expectedRange.getExtent()) &&
                  sameScalarExpression(predicateRange.getStep(),
                                       expectedRange.getStep());
    if (!sameRange)
      return false;
    if (comparison.getPredicate() == ComparePredicate::Ge)
      return integerConstant(rhs) == 0;
    return sameTailEnd(rhs, expectedEnd);
  });
}

PhysicalAccessFootprint
PhysicalProgramAnalysis::footprint(Operation *access) {
  PhysicalAccessFootprint result;
  auto collect = [&](Value resource, ValueRange coordinates,
                     ArrayRef<int64_t> sourceAxes, Value validity, Value fill) {
    result.resource = resource;
    result.coordinates.append(coordinates.begin(), coordinates.end());
    result.sourceAxes.append(sourceAxes.begin(), sourceAxes.end());
    result.validity = validity;
    result.fill = fill;
    result.state = resource && coordinates.size() == sourceAxes.size()
                       ? PhysicalFactState::Exact
                       : PhysicalFactState::Unknown;
    result.rangeState = PhysicalFactState::Exact;
    for (Value coordinate : coordinates) {
      PhysicalRangeFact ranges;
      ranges.state = PhysicalFactState::Exact;
      SmallPtrSet<Operation *, 32> visited;
      // A scalar coordinate may depend on a completed reduction, but that
      // reduction's input lanes are not lanes of this memory access.  Keep the
      // coordinate SSA dependency without applying its ancestors' tail masks.
      collectRanges(coordinate, std::nullopt, ranges, visited,
                    /*followScalarDependencies=*/false);
      // A footprint records the complete set of ranges, not a request for one
      // unique range.  Multiple roots are therefore exact here.  A coordinate
      // with no range roots is also exact when it is a scalar/broadcast-only
      // expression.  Only an operation that blocks provenance makes the
      // address footprint unknown.
      if (!ranges.blockers.empty())
        result.rangeState = PhysicalFactState::Unknown;
      for (Operation *blocker : ranges.blockers)
        appendUnique(result.blockers, blocker);
      for (MakeRangeOp range : ranges.roots)
        appendUnique(result.ranges, range);
    }
  };
  if (auto relation = dyn_cast<AccessOpInterface>(access))
    collect(relation.getAccessResource(), relation.getAccessCoordinates(),
            relation.getAccessSourceAxes(), relation.getAccessValidity(),
            relation.getAccessFill());
  else
    result.state = PhysicalFactState::Unknown;
  return result;
}

bool haveDisjointPrivateBufferAccesses(Operation *lhs, Operation *rhs) {
  if (!isa<LoadOp, StoreOp>(lhs) || !isa<LoadOp, StoreOp>(rhs))
    return false;
  PhysicalProgramAnalysis analysis(lhs->getParentOfType<func::FuncOp>());
  auto left = analysis.footprint(lhs);
  auto right = analysis.footprint(rhs);
  if (left.state != PhysicalFactState::Exact ||
      right.state != PhysicalFactState::Exact)
    return false;
  auto buffer = dyn_cast<BufferType>(left.resource.getType());
  if (!buffer || buffer.getScope().getValue() != BufferScope::ProgramPrivate ||
      buffer.isInvocationWorkspace() || !left.resource.getDefiningOp<BufferOp>())
    return false;
  if (left.resource != right.resource) {
    ResourceAliasAnalysis aliases;
    return aliases.alias(left.resource, right.resource).isNo();
  }

  auto scalar = [](Value value) {
    value = stripBroadcast(value);
    return value.getType().isIntOrIndex() ? stripScalarIdentity(value) : Value();
  };
  auto laterIteration = [&](Value coordinate, Value point) {
    auto argument = dyn_cast_or_null<BlockArgument>(coordinate);
    auto loop = argument
                    ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp())
                    : scf::ForOp();
    if (!loop || argument != loop.getInductionVar() ||
        integerConstant(loop.getStep()) != 1)
      return false;
    auto lower = stripScalarIdentity(loop.getLowerBound()).getDefiningOp<BinaryOp>();
    if (!lower || lower.getOperatorKind() != BinaryOperator::Add)
      return false;
    for (auto [base, offset] :
         {std::pair{lower.getLhs(), lower.getRhs()},
          std::pair{lower.getRhs(), lower.getLhs()}}) {
      auto amount = integerConstant(offset);
      if (stripScalarIdentity(base) != point || !amount || *amount <= 0)
        continue;
      IndexBounds pointBounds = queryIndexBounds(point);
      IndexBounds iterationBounds = queryIndexBounds(coordinate);
      auto kernel = lhs->getParentOfType<func::FuncOp>();
      auto pointExtent = pointBounds.nonNegative && pointBounds.upper
                             ? nonNegativeExtentBounds(kernel, pointBounds.upper)
                             : std::nullopt;
      auto iterationExtent = iterationBounds.nonNegative && iterationBounds.upper
                                 ? nonNegativeExtentBounds(kernel, iterationBounds.upper)
                                 : std::nullopt;
      // The positive unit-step loop starts strictly after the point. Both
      // its start expression and final increment must remain representable.
      if (pointExtent && iterationExtent &&
          pointExtent->second <= std::numeric_limits<int64_t>::max() - *amount &&
          iterationExtent->second < std::numeric_limits<int64_t>::max())
        return true;
    }
    return false;
  };
  auto excludes = [&](const PhysicalAccessFootprint &pointAccess,
                      const PhysicalAccessFootprint &other) {
    for (auto [index, pointCoordinate] : llvm::enumerate(pointAccess.coordinates)) {
      Value point = scalar(pointCoordinate);
      if (!point)
        continue;
      auto axis = llvm::find(other.sourceAxes, pointAccess.sourceAxes[index]);
      if (axis == other.sourceAxes.end())
        continue;
      Value coordinate = other.coordinates[axis - other.sourceAxes.begin()];
      if (laterIteration(scalar(coordinate), point))
        return true;
      if (!other.validity)
        continue;
      auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
      auto predicateType = dyn_cast<FragmentType>(other.validity.getType());
      if (coordinateType) {
        if (!predicateType || coordinateType.getOwner() != predicateType.getOwner() ||
            !queryBroadcastProjection(coordinateType, predicateType).isExact())
          continue;
        // A repeated source axis could place the predicate and address on
        // different Cartesian occurrences. Require a unique target occurrence.
        bool unique = llvm::all_of(coordinateType.getAxisMaps(), [&](Attribute attr) {
          auto source = cast<AxisMapAttr>(attr);
          return llvm::count_if(predicateType.getAxisMaps(), [&](Attribute target) {
            auto mapping = cast<AxisMapAttr>(target);
            return sourceAxisIdentity(mapping) == sourceAxisIdentity(source) &&
                   mapping.getDimensionId() == source.getDimensionId();
          }) == 1;
        });
        if (!unique)
          continue;
      }
      std::function<bool(Value)> inspect = [&](Value predicate) {
        if (auto broadcast = predicate.getDefiningOp<BroadcastOp>()) {
          if (auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
              input && !queryBroadcastProjection(
                           input, cast<FragmentType>(predicate.getType())).isExact())
            return false;
          return inspect(broadcast.getValue());
        }
        if (auto splat = predicate.getDefiningOp<SplatOp>())
          return inspect(splat.getValue());
        if (auto conjunction = predicate.getDefiningOp<BinaryOp>();
            conjunction && conjunction.getOperatorKind() == BinaryOperator::LogicalAnd)
          return inspect(conjunction.getLhs()) || inspect(conjunction.getRhs());
        auto compare = predicate.getDefiningOp<CompareOp>();
        if (!compare || (compare.getPredicate() != ComparePredicate::Ne &&
                         compare.getPredicate() != ComparePredicate::Lt &&
                         compare.getPredicate() != ComparePredicate::Gt))
          return false;
        return (compare.getLhs() == coordinate && scalar(compare.getRhs()) == point) ||
               (compare.getRhs() == coordinate && scalar(compare.getLhs()) == point);
      };
      if (inspect(other.validity))
        return true;
    }
    return false;
  };
  return excludes(left, right) || excludes(right, left);
}

PhysicalAccessBoundaryFact
PhysicalProgramAnalysis::boundaryValidity(Operation *access,
                                          bool allowRangeGuards) {
  PhysicalAccessBoundaryFact result;
  PhysicalAccessFootprint accessFact = footprint(access);
  result.blockers = accessFact.blockers;
  auto view = accessFact.resource
                  ? dyn_cast<ViewType>(accessFact.resource.getType())
                  : ViewType();
  if (accessFact.state != PhysicalFactState::Exact ||
      accessFact.rangeState != PhysicalFactState::Exact || !view) {
    if (result.blockers.empty())
      appendUnique(result.blockers, access);
    return result;
  }
  if (!accessFact.validity) {
    result.state = PhysicalFactState::Exact;
    return result;
  }

  llvm::DenseSet<int64_t> boundaryAxes;
  std::optional<bool> boundedMembers;
  std::function<PhysicalFactState(Value)> analyze =
      [&](Value value) -> PhysicalFactState {
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
      return analyze(broadcast.getValue());
    if (auto splat = value.getDefiningOp<SplatOp>())
      return analyze(splat.getValue());
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      return analyze(reshape.getValue());
    if (auto transpose = value.getDefiningOp<TransposeOp>())
      return analyze(transpose.getValue());
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      return integer && integer.getType().isInteger(1) &&
                     integer.getValue().isOne()
                 ? PhysicalFactState::Exact
                 : PhysicalFactState::Unknown;
    }
    if (auto conjunction = value.getDefiningOp<BinaryOp>()) {
      Type element = conjunction.getResult().getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      bool logical =
          conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
      bool bitwiseI1 =
          conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
          element.isInteger(1);
      bool disjunction =
          conjunction.getOperatorKind() == BinaryOperator::LogicalOr ||
          (conjunction.getOperatorKind() == BinaryOperator::BitwiseOr &&
           element.isInteger(1));
      if (disjunction) {
        // Only this implication preserves the exact native boundary mask:
        // (upper(coordinate) < bound) OR (coordinate < bound). Merely proving
        // each arm safe would not make an arbitrary OR a conjunction of axes.
        for (auto [directValue, proofValue] :
             {std::pair{conjunction.getLhs(), conjunction.getRhs()},
              std::pair{conjunction.getRhs(), conjunction.getLhs()}}) {
          auto direct = directValue.getDefiningOp<CompareOp>();
          auto proof = proofValue.getDefiningOp<CompareOp>();
          if (!direct || !proof ||
              direct.getPredicate() != ComparePredicate::Lt ||
              proof.getPredicate() != ComparePredicate::Lt)
            continue;
          auto directLimit = queryLaunchExpression(direct.getRhs());
          auto proofLimit = queryLaunchExpression(proof.getRhs());
          if (!samePhysicalScalarExpression(direct.getRhs(), proof.getRhs()) &&
              !(directLimit && proofLimit && directLimit == proofLimit))
            continue;
          for (Value coordinate : accessFact.coordinates)
            if (derivesFromAccessCoordinate(direct.getLhs(), coordinate) &&
                isInclusiveCoordinateUpperBound(proof.getLhs(), coordinate))
              return analyze(directValue);
        }
        appendUnique(result.blockers, conjunction);
        return PhysicalFactState::Unknown;
      }
      if (!logical && !bitwiseI1) {
        appendUnique(result.blockers, conjunction);
        return PhysicalFactState::Unknown;
      }
      PhysicalFactState lhs = analyze(conjunction.getLhs());
      PhysicalFactState rhs = analyze(conjunction.getRhs());
      if (lhs == PhysicalFactState::Ambiguous ||
          rhs == PhysicalFactState::Ambiguous)
        return PhysicalFactState::Ambiguous;
      return lhs == PhysicalFactState::Exact &&
                     rhs == PhysicalFactState::Exact
                 ? PhysicalFactState::Exact
                 : PhysicalFactState::Unknown;
    }
    auto comparison = value.getDefiningOp<CompareOp>();
    bool upperComparison =
        comparison && comparison.getPredicate() == ComparePredicate::Lt;
    bool lowerComparison =
        comparison && comparison.getPredicate() == ComparePredicate::Ge &&
        integerConstant(comparison.getRhs()) == 0;
    if (comparison && comparison.getPredicate() == ComparePredicate::Ge &&
        !lowerComparison) {
      Value lower = stripScalarIdentity(comparison.getRhs());
      for (MakeRangeOp range : accessFact.ranges) {
        if (!isUnitStepRange(range) || !lower.getType().isIndex() ||
            !derivesFromAccessCoordinate(comparison.getLhs(), range.getResult()) ||
            !samePhysicalScalarExpression(lower, range.getLogicalStart()) ||
            !samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()) ||
            !IndexRelations().positive(range.getExtent()))
          continue;
        auto start = queryIntegerRange(range.getStart());
        auto extent = queryIntegerRange(range.getExtent());
        bool noWrap = start && extent && extent->smin().isStrictlyPositive() &&
            provesSignedNoWrap(BinaryOperator::Add, *start,
                ConstantIntRanges::fromSigned(APInt(64, 0),
                                               extent->smax() - 1));
        noWrap |= IndexRelations().alignedUnitWindow(range.getStart(), range.getExtent());
        if (!noWrap && !allowRangeGuards)
          continue;
        if (!noWrap) {
          // The native consumer must establish a complete nonnegative tile:
          // stop - start >= extent, with both endpoints nonnegative. Then
          // start + (extent - 1) < stop cannot wrap, so every physical member
          // satisfies this lower predicate. Partial tiles keep their original
          // predicated access; start == logicalStart alone is insufficient.
          std::pair<MakeRangeOp, Value> bound{range, range.getLogicalStop()};
          if (!llvm::is_contained(result.rangeBounds, bound))
            result.rangeBounds.push_back(bound);
        }
        return PhysicalFactState::Exact;
      }
    }
    if (!upperComparison && !lowerComparison) {
      appendUnique(result.blockers, value.getDefiningOp());
      return PhysicalFactState::Unknown;
    }

    std::optional<int64_t> matchedAxis;
    bool requiresBoundary = false;
    for (auto [coordinate, sourceAxis] :
         llvm::zip(accessFact.coordinates, accessFact.sourceAxes)) {
      if (!derivesFromAccessCoordinate(comparison.getLhs(), coordinate))
        continue;
      bool worksetViewBoundary = false;
      if (upperComparison && comparison->hasAttr(physicalTailAttr)) {
        Value accessCoordinate = stripIntegerIndexCasts(coordinate);
        auto range = accessCoordinate.getDefiningOp<MakeRangeOp>();
        auto workset =
            range ? range.getStart().getDefiningOp<WorksetCoordinateOp>()
                  : WorksetCoordinateOp();
        auto accessView = dyn_cast<ViewType>(accessFact.resource.getType());
        if (range && range->hasAttr(worksetCoordinateRangeAttr) &&
            !range->hasAttr(sourceSubregionAttr) && workset && accessView &&
            sourceAxisIdentity(range) ==
                PhysicalSourceAxis{workset.getSourceId(),
                                   workset.getSourceAxis(), false} &&
            sourceAxis >= 0 &&
            sourceAxis < static_cast<int64_t>(accessView.getRank())) {
          ArrayRef<int64_t> dimensions =
              accessView.getLayout().getDimensionIds().asArrayRef();
          worksetViewBoundary =
              dimensions[sourceAxis] ==
              static_cast<int64_t>(workset.getDimensionId());
        }
      }
      bool viewBoundary =
          lowerComparison || worksetViewBoundary ||
          matchesResourceExtent(comparison.getRhs(), accessFact.resource,
                                sourceAxis);
      bool exactRange =
          upperComparison && hasExactPhysicalRangeCoverage(
                                 coordinate, comparison.getRhs());
      if (upperComparison && !viewBoundary && !exactRange &&
          capacityCoversResourceExtent(comparison.getRhs(), accessFact.resource,
                                      sourceAxis)) {
        // A padded capacity is a weaker bound than the view extent. It is
        // redundant only when the original validity (or an unconditional
        // range fact) already confines active members to that view.
        if (!boundedMembers)
          boundedMembers = accessBounds(access).isExact();
        exactRange = *boundedMembers;
      }
      MakeRangeOp guardedRange;
      Value rangeBound;
      if (allowRangeGuards && upperComparison && !viewBoundary && !exactRange) {
        auto range = stripIntegerIndexCasts(coordinate)
                         .getDefiningOp<MakeRangeOp>();
        Value bound = stripScalarIdentity(comparison.getRhs());
        if (range && isUnitStepRange(range) && bound.getType().isIndex() &&
            sameScalarExpression(bound, range.getLogicalStop())) {
          guardedRange = range;
          rangeBound = bound;
        }
      }
      if (!viewBoundary && !exactRange && !guardedRange)
        continue;
      if (matchedAxis) {
        appendUnique(result.blockers, comparison);
        return PhysicalFactState::Ambiguous;
      }
      matchedAxis = sourceAxis;
      if (guardedRange &&
          !llvm::is_contained(result.rangeBounds,
                             std::make_pair(guardedRange, rangeBound)))
        result.rangeBounds.emplace_back(guardedRange, rangeBound);
      requiresBoundary =
          viewBoundary && !coordinateRangeWithinResource(
                              coordinate, accessFact.resource, sourceAxis);
    }
    if (!matchedAxis) {
      // A component range of a composed address may have its own logical tail
      // (for example group*width+channel). A whole-range guard discharges that
      // predicate without equating it to a boundary of the resource axis.
      if (allowRangeGuards)
        for (MakeRangeOp range : accessFact.ranges) {
          Value bound = stripScalarIdentity(range.getLogicalStop());
          if (!isUnitStepRange(range) || !bound.getType().isIndex() ||
              !isTailPredicate(value, {{range, bound}}))
            continue;
          if (!llvm::is_contained(result.rangeBounds,
                                  std::make_pair(range, bound)))
            result.rangeBounds.emplace_back(range, bound);
          return PhysicalFactState::Exact;
        }
      appendUnique(result.blockers, comparison);
      return PhysicalFactState::Unknown;
    }
    if (requiresBoundary)
      boundaryAxes.insert(*matchedAxis);
    return PhysicalFactState::Exact;
  };

  result.state = analyze(accessFact.validity);
  if (result.state == PhysicalFactState::Exact) {
    result.boundaryAxes.append(boundaryAxes.begin(), boundaryAxes.end());
    llvm::sort(result.boundaryAxes);
  }
  return result;
}

PhysicalAccessBoundsFact
PhysicalProgramAnalysis::accessBounds(Operation *access) {
  PhysicalAccessBoundsFact result;
  PhysicalAccessFootprint accessFact = footprint(access);
  result.blockers = accessFact.blockers;
  if (accessFact.state != PhysicalFactState::Exact || !accessFact.resource ||
      accessFact.coordinates.size() != accessFact.sourceAxes.size()) {
    appendUnique(result.blockers, access);
    return result;
  }

  unsigned rank = 0;
  if (auto view = dyn_cast<ViewType>(accessFact.resource.getType()))
    rank = view.getRank();
  else if (auto buffer = dyn_cast<BufferType>(accessFact.resource.getType()))
    rank = buffer.getShape().size();
  else if (auto fragment =
               dyn_cast<FragmentType>(accessFact.resource.getType()))
    rank = fragment.getShape().size();
  if (rank == 0 && !accessFact.coordinates.empty()) {
    appendUnique(result.blockers, access);
    return result;
  }

  struct AxisBounds {
    bool lower = false;
    bool upper = false;
  };
  SmallVector<AxisBounds> bounds(rank);
  SmallVector<bool> required(rank, false);
  for (auto [coordinate, sourceAxis] :
       llvm::zip(accessFact.coordinates, accessFact.sourceAxes)) {
    if (sourceAxis < 0 || sourceAxis >= static_cast<int64_t>(rank)) {
      appendUnique(result.blockers, access);
      return result;
    }
    required[sourceAxis] = true;
    if (coordinateRangeWithinResource(coordinate, accessFact.resource,
                                      sourceAxis)) {
      bounds[sourceAxis].lower = true;
      bounds[sourceAxis].upper = true;
    } else if (IndexRelations().nonnegative(coordinate)) {
      bounds[sourceAxis].lower = true;
    }
  }

  DominanceInfo dominance(kernel);
  kernel.walk([&](AssumeInBoundsOp assumption) {
    if (assumption.getResource() != accessFact.resource ||
        assumption.getAxis() >= rank ||
        !dominance.properlyDominates(assumption.getOperation(), access))
      return;
    for (auto [coordinate, sourceAxis] :
         llvm::zip(accessFact.coordinates, accessFact.sourceAxes)) {
      if (sourceAxis != static_cast<int64_t>(assumption.getAxis()) ||
          !derivesFromAccessCoordinate(assumption.getIndex(), coordinate))
        continue;
      bounds[sourceAxis].lower = true;
      bounds[sourceAxis].upper = true;
      if (!llvm::is_contained(result.assumedAxes, sourceAxis))
        result.assumedAxes.push_back(sourceAxis);
    }
  });

  std::function<bool(Value)> analyze = [&](Value value) -> bool {
    if (!value)
      return false;
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
      return analyze(broadcast.getValue());
    if (auto splat = value.getDefiningOp<SplatOp>())
      return analyze(splat.getValue());
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      return analyze(reshape.getValue());
    if (auto transpose = value.getDefiningOp<TransposeOp>())
      return analyze(transpose.getValue());
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      return integer && integer.getType().isInteger(1) &&
             integer.getValue().isZero();
    }
    if (auto conjunction = value.getDefiningOp<BinaryOp>()) {
      Type element = conjunction.getResult().getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      bool logical =
          conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
      bool bitwiseI1 =
          conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
          element.isInteger(1);
      if (logical || bitwiseI1) {
        bool lhsInactive = analyze(conjunction.getLhs());
        bool rhsInactive = analyze(conjunction.getRhs());
        return lhsInactive || rhsInactive;
      }
      bool disjunction =
          conjunction.getOperatorKind() == BinaryOperator::LogicalOr ||
          (conjunction.getOperatorKind() == BinaryOperator::BitwiseOr &&
           element.isInteger(1));
      if (disjunction) {
        // Each active arm must establish a bound. Facts from an enclosing
        // conjunction apply to both arms; facts discovered in one arm do not.
        auto before = bounds;
        bool lhsInactive = analyze(conjunction.getLhs());
        auto left = bounds;
        bounds = before;
        bool rhsInactive = analyze(conjunction.getRhs());
        for (unsigned axis = 0; axis < rank; ++axis) {
          bounds[axis].lower = before[axis].lower ||
              ((lhsInactive || left[axis].lower) &&
               (rhsInactive || bounds[axis].lower));
          bounds[axis].upper = before[axis].upper ||
              ((lhsInactive || left[axis].upper) &&
               (rhsInactive || bounds[axis].upper));
        }
        return lhsInactive && rhsInactive;
      }
      return false;
    }
    auto comparison = value.getDefiningOp<CompareOp>();
    if (!comparison)
      return false;
    for (auto [coordinate, sourceAxis] :
         llvm::zip(accessFact.coordinates, accessFact.sourceAxes)) {
      if (sourceAxis < 0 || sourceAxis >= static_cast<int64_t>(rank))
        continue;
      bool lhsCoordinate =
          derivesFromAccessCoordinate(comparison.getLhs(), coordinate);
      bool rhsCoordinate =
          derivesFromAccessCoordinate(comparison.getRhs(), coordinate);
      if ((comparison.getPredicate() == ComparePredicate::Ge &&
           lhsCoordinate && integerConstant(comparison.getRhs()) == 0) ||
          (comparison.getPredicate() == ComparePredicate::Le &&
           rhsCoordinate && integerConstant(comparison.getLhs()) == 0))
        bounds[sourceAxis].lower = true;
      if ((comparison.getPredicate() == ComparePredicate::Lt &&
           (lhsCoordinate || isInclusiveCoordinateUpperBound(
                                 comparison.getLhs(), coordinate)) &&
           upperBoundWithinResource(
                                comparison.getRhs(), accessFact.resource,
                                sourceAxis)) ||
          (comparison.getPredicate() == ComparePredicate::Gt &&
           (rhsCoordinate || isInclusiveCoordinateUpperBound(
                                 comparison.getRhs(), coordinate)) &&
           upperBoundWithinResource(
                                comparison.getLhs(), accessFact.resource,
                                sourceAxis)))
        bounds[sourceAxis].upper = true;
    }
    return false;
  };

  bool alwaysInactive = analyze(accessFact.validity);
  for (unsigned axis = 0; axis < rank; ++axis) {
    if (!required[axis] || alwaysInactive)
      continue;
    if (!bounds[axis].lower)
      result.missingLowerAxes.push_back(axis);
    if (!bounds[axis].upper)
      result.missingUpperAxes.push_back(axis);
    if (!bounds[axis].lower || !bounds[axis].upper)
      result.unprovenAxes.push_back(axis);
  }
  if (result.unprovenAxes.empty()) {
    result.state = PhysicalFactState::Exact;
    return result;
  }
  appendUnique(result.blockers,
               accessFact.validity ? accessFact.validity.getDefiningOp()
                                   : access);
  return result;
}

} // namespace intent::gpu
