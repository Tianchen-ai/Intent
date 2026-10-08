#include "ScalarExpressions.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu {
using namespace detail;

std::optional<int64_t> constantLogicalRangeCardinality(MakeRangeOp range) {
  auto constant = [](Value value) -> std::optional<int64_t> {
    auto folded = dyn_cast_or_null<IntegerAttr>(
        UniformValueAnalysis(describeUniformValue).evaluate(value));
    return folded && folded.getValue().getBitWidth() <= 64
               ? std::optional<int64_t>(folded.getInt()) : std::nullopt;
  };
  auto step = constant(range.getStep());
  if (!step || *step <= 0)
    return std::nullopt;
  std::optional<__int128> distance;
  auto start = constant(range.getLogicalStart());
  auto stop = constant(range.getLogicalStop());
  if (start && stop)
    distance = static_cast<__int128>(*stop) - *start;
  else if (auto add = stripScalarIdentity(range.getLogicalStop())
                          .getDefiningOp<BinaryOp>();
           add && add.getOperatorKind() == BinaryOperator::Add)
    for (auto [base, offset] : {std::pair{add.getLhs(), add.getRhs()},
                               std::pair{add.getRhs(), add.getLhs()}})
      if (samePhysicalScalarExpression(base, range.getLogicalStart()))
        if (auto size = constant(offset))
          distance = *size;
  if (!distance) {
    struct Offset {
      Value base;
      __int128 amount = 0;
      __int128 minimum = 0;
      __int128 maximum = 0;
      __int128 scale = 1;
    };
    std::function<Offset(Value)> splitOffset = [&](Value value) -> Offset {
      value = stripScalarIdentity(value);
      auto binary = value.getDefiningOp<BinaryOp>();
      if (!binary)
        return {value};
      Value base;
      std::optional<__int128> increment;
      if (binary.getOperatorKind() == BinaryOperator::Multiply) {
        auto factor = constant(binary.getRhs());
        base = binary.getLhs();
        if (!factor) {
          factor = constant(binary.getLhs());
          base = binary.getRhs();
        }
        if (!factor || *factor <= 0)
          return {value};
        Offset result = splitOffset(base);
        const __int128 lower = std::numeric_limits<int64_t>::min();
        const __int128 upper = std::numeric_limits<int64_t>::max();
        if (result.scale > upper / *factor || result.minimum < lower ||
            result.maximum > upper)
          return {value};
        result.scale *= *factor;
        result.amount *= *factor;
        result.minimum = std::min(result.minimum, result.minimum * *factor);
        result.maximum = std::max(result.maximum, result.maximum * *factor);
        return result;
      }
      if (binary.getOperatorKind() == BinaryOperator::Add) {
        if (auto rhs = constant(binary.getRhs())) {
          base = binary.getLhs();
          increment = *rhs;
        } else if (auto lhs = constant(binary.getLhs())) {
          base = binary.getRhs();
          increment = *lhs;
        }
      } else if (binary.getOperatorKind() == BinaryOperator::Subtract) {
        if (auto rhs = constant(binary.getRhs())) {
          base = binary.getLhs();
          increment = -static_cast<__int128>(*rhs);
        }
      }
      if (!increment)
        return {value};
      Offset result = splitOffset(base);
      result.amount += *increment;
      result.minimum = std::min(result.minimum, result.amount);
      result.maximum = std::max(result.maximum, result.amount);
      return result;
    };
    Offset begin = splitOffset(range.getLogicalStart());
    Offset end = splitOffset(range.getLogicalStop());
    if (begin.scale == end.scale &&
        samePhysicalScalarExpression(begin.base, end.base)) {
      auto upper = queryNonNegativeIndexUpperBound(begin.base);
      // Cancel a common affine base only when all intermediate index arithmetic
      // stays in range; signed wrap must not turn a short slice into a full one.
      if (upper && upper.getKind() ==
                       PhysicalExprKind::Constant &&
          std::min(begin.minimum, end.minimum) >=
              std::numeric_limits<int64_t>::min() &&
          std::max(begin.maximum, end.maximum) <=
              std::numeric_limits<int64_t>::max() &&
          static_cast<__int128>(upper.getValue()) * begin.scale +
                  std::max(begin.maximum, end.maximum) <=
              std::numeric_limits<int64_t>::max())
        distance = end.amount - begin.amount;
    }
  }
  if (!distance || *distance < 0)
    return std::nullopt;
  __int128 size = (*distance + *step - 1) / *step;
  return size <= std::numeric_limits<int64_t>::max()
             ? std::optional<int64_t>(size) : std::nullopt;
}

bool isProvablySingletonLogicalRange(MakeRangeOp range) {
  if (auto size = constantLogicalRangeCardinality(range))
    return *size == 1;

  Value logicalStop = stripScalarIdentity(range.getLogicalStop());
  auto add = logicalStop.getDefiningOp<BinaryOp>();
  if (!add || add.getOperatorKind() != BinaryOperator::Add)
    return false;
  return (sameScalarExpression(add.getLhs(), range.getLogicalStart()) &&
          sameScalarExpression(add.getRhs(), range.getStep())) ||
         (sameScalarExpression(add.getRhs(), range.getLogicalStart()) &&
          sameScalarExpression(add.getLhs(), range.getStep()));
}

PhysicalSourceAxis sourceAxisIdentity(AxisMapAttr mapping) {
  return {mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived()};
}

PhysicalSourceAxis sourceAxisIdentity(MakeRangeOp range) {
  return {range.getSourceId(), range.getSourceAxis(), range.getDerived()};
}

PhysicalAxisProjection queryFragmentAxis(Type type,
                                         PhysicalSourceAxis source,
                                         std::optional<int64_t> expectedDimension) {
  PhysicalAxisProjection result;
  result.source = source;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return result;
  std::optional<unsigned> axis;
  std::optional<int64_t> dimension;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getSourceId() != source.sourceId ||
        mapping.getSourceAxis() != source.sourceAxis ||
        mapping.getDerived() != source.derived ||
        (expectedDimension && mapping.getDimensionId() != *expectedDimension))
      continue;
    if ((axis && *axis != mapping.getFragmentAxis()) ||
        (dimension && *dimension != mapping.getDimensionId())) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    axis = mapping.getFragmentAxis();
    dimension = mapping.getDimensionId();
  }
  if (!axis || !dimension)
    return result;
  result.state = PhysicalFactState::Exact;
  result.fragmentAxis = *axis;
  result.dimensionId = *dimension;
  return result;
}

FailureOr<AxisMapAttr> queryAxisMap(Type type, unsigned fragmentAxis) {
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return failure();
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getFragmentAxis() == fragmentAxis)
      return mapping;
  }
  return failure();
}

FailureOr<int64_t> queryRangeDimension(MakeRangeOp range) {
  return querySourceDimension(range.getResult().getType(),
                              sourceAxisIdentity(range));
}

bool sameLogicalRange(MakeRangeOp lhs, MakeRangeOp rhs) {
  if (!lhs || !rhs || lhs.getSourceId() != rhs.getSourceId() ||
      lhs.getSourceAxis() != rhs.getSourceAxis() ||
      lhs.getDerived() != rhs.getDerived())
    return false;
  return samePhysicalScalarExpression(lhs.getLogicalStart(), rhs.getLogicalStart()) &&
         samePhysicalScalarExpression(lhs.getLogicalStop(), rhs.getLogicalStop()) &&
         samePhysicalScalarExpression(lhs.getStep(), rhs.getStep());
}

bool isUnitStepRange(MakeRangeOp range) {
  return range && isUnitStepValue(range.getStep());
}

FailureOr<int64_t> querySubregionParentDimension(MakeRangeOp range) {
  if (!range)
    return failure();
  auto parent = range->getAttrOfType<IntegerAttr>(sourceSubregionAttr);
  return parent && parent.getInt() > 0
             ? FailureOr<int64_t>(parent.getInt())
             : FailureOr<int64_t>(failure());
}

FailureOr<MakeRangeOp>
queryExactLogicalRange(const PhysicalRangeFact &fact) {
  if (fact.state == PhysicalFactState::Unknown || fact.roots.empty())
    return failure();
  MakeRangeOp first = fact.roots.front();
  return llvm::all_of(fact.roots, [&](MakeRangeOp range) {
           return sameLogicalRange(first, range);
         })
             ? FailureOr<MakeRangeOp>(first)
             : FailureOr<MakeRangeOp>(failure());
}

SmallVector<PhysicalAxisProjection, 2>
queryFragmentAxes(Type type, PhysicalSourceAxis source) {
  SmallVector<PhysicalAxisProjection, 2> results;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return results;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getSourceId() != source.sourceId ||
        mapping.getSourceAxis() != source.sourceAxis ||
        mapping.getDerived() != source.derived)
      continue;
    results.push_back(PhysicalAxisProjection{
        PhysicalFactState::Exact, source, mapping.getDimensionId(),
        mapping.getFragmentAxis()});
  }
  return results;
}

SmallVector<PhysicalAxisProjection, 2>
queryRangeProjections(Type type, MakeRangeOp range) {
  SmallVector<PhysicalAxisProjection, 2> sourceResults =
      queryFragmentAxes(type, sourceAxisIdentity(range));
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  if (succeeded(dimension)) {
    SmallVector<PhysicalAxisProjection, 2> dimensionResults = sourceResults;
    llvm::erase_if(dimensionResults,
                   [&](const PhysicalAxisProjection &projection) {
      return projection.dimensionId != *dimension;
    });
    if (!dimensionResults.empty())
      return dimensionResults;
    PhysicalDimensionProjection projection =
        queryFragmentDimension(type, *dimension);
    if (projection.isExact())
      return {PhysicalAxisProjection{
          PhysicalFactState::Exact, sourceAxisIdentity(range), *dimension,
          projection.fragmentAxis}};
  }
  return sourceResults.size() == 1
             ? sourceResults
             : SmallVector<PhysicalAxisProjection, 2>{};
}

PhysicalDimensionProjection queryFragmentDimension(Type type,
                                                   int64_t dimensionId) {
  PhysicalDimensionProjection result;
  result.dimensionId = dimensionId;
  if (dimensionId <= 0)
    return result;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return result;
  std::optional<unsigned> axis;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getDimensionId() != dimensionId)
      continue;
    if (axis && *axis != mapping.getFragmentAxis()) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    axis = mapping.getFragmentAxis();
  }
  if (!axis)
    return result;
  result.state = PhysicalFactState::Exact;
  result.fragmentAxis = *axis;
  return result;
}

SmallVector<PhysicalDimensionProjection, 2>
queryFragmentDimensions(Type type, int64_t dimensionId) {
  SmallVector<PhysicalDimensionProjection, 2> results;
  if (dimensionId <= 0)
    return results;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return results;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getDimensionId() != dimensionId)
      continue;
    results.push_back(PhysicalDimensionProjection{
        PhysicalFactState::Exact, dimensionId, mapping.getFragmentAxis()});
  }
  return results;
}

FailureOr<int64_t> querySourceDimension(Type type, PhysicalSourceAxis source) {
  if (auto range = dyn_cast<RangeType>(type))
    return range.getSourceId() == source.sourceId &&
                   range.getSourceAxis() == source.sourceAxis &&
                   range.getDerived() == source.derived &&
                   range.getDimensionId() > 0
               ? FailureOr<int64_t>(range.getDimensionId())
               : FailureOr<int64_t>(failure());
  PhysicalAxisProjection projection = queryFragmentAxis(type, source);
  return projection.isExact() && projection.dimensionId > 0
             ? FailureOr<int64_t>(projection.dimensionId)
             : FailureOr<int64_t>(failure());
}

PhysicalAxisProjection
queryCoordinateIndex(ValueRange coordinates, PhysicalSourceAxis source,
                     std::optional<int64_t> dimension) {
  PhysicalAxisProjection result;
  result.source = source;
  std::optional<unsigned> coordinateIndex;
  for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
    PhysicalAxisProjection projection =
        queryFragmentAxis(coordinate.getType(), source, dimension);
    if (projection.state == PhysicalFactState::Ambiguous) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    if (!projection.isExact())
      continue;
    if (coordinateIndex) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    coordinateIndex = index;
  }
  if (!coordinateIndex)
    return result;
  result.state = PhysicalFactState::Exact;
  PhysicalAxisProjection projection =
      queryFragmentAxis(coordinates[*coordinateIndex].getType(), source, dimension);
  result.dimensionId = projection.dimensionId;
  result.fragmentAxis = *coordinateIndex;
  return result;
}

FailureOr<unsigned> queryCoordinatePosition(ValueRange coordinates,
                                            PhysicalSourceAxis source) {
  PhysicalAxisProjection result = queryCoordinateIndex(coordinates, source);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
}

FailureOr<unsigned>
PhysicalProgramAnalysis::fragmentAxis(Type type,
                                      PhysicalSourceAxis source) const {
  PhysicalAxisProjection result = queryFragmentAxis(type, source);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
}

FailureOr<unsigned> PhysicalProgramAnalysis::accessCoordinatePosition(
    LoadOp load, AxisMapAttr mapping, Value operand) {
  if (PhysicalAxisProjection direct = queryCoordinateIndex(
          load.getCoordinates(), sourceAxisIdentity(mapping),
          mapping.getDimensionId());
      direct.isExact())
    return direct.fragmentAxis;
  FailureOr<MakeRangeOp> selected = queryExactLogicalRange(
      axisRanges(operand, mapping.getFragmentAxis()));
  std::optional<unsigned> replayed;
  for (auto [position, coordinate] : llvm::enumerate(load.getCoordinates())) {
    PhysicalAxisProjection axis = queryFragmentAxis(
        coordinate.getType(), sourceAxisIdentity(mapping),
        mapping.getDimensionId());
    FailureOr<MakeRangeOp> coordinateRange = queryExactLogicalRange(
        axis.isExact() ? axisRanges(coordinate, axis.fragmentAxis)
                       : sourceRanges(coordinate, sourceAxisIdentity(mapping)));
    if (failed(coordinateRange) ||
        (succeeded(selected) && !sameLogicalRange(*selected, *coordinateRange)))
      continue;
    if (replayed)
      return failure();
    replayed = position;
  }
  if (replayed)
    return *replayed;
  auto result = dyn_cast<FragmentType>(load.getResult().getType());
  if (result && result.getShape().size() == load.getCoordinates().size() &&
      llvm::all_of(load.getCoordinates(), [](Value coordinate) {
        auto fragment = dyn_cast<FragmentType>(coordinate.getType());
        return fragment && fragment.getShape().size() == 1;
      }) &&
      mapping.getFragmentAxis() < load.getCoordinates().size())
    return mapping.getFragmentAxis();
  auto view = dyn_cast<ViewType>(load.getResource().getType());
  if (!view || mapping.getDimensionId() <= 0)
    return failure();
  std::optional<unsigned> resourceAxis;
  for (auto [axis, dimension] :
       llvm::enumerate(view.getLayout().getDimensionIds().asArrayRef())) {
    if (dimension != mapping.getDimensionId())
      continue;
    if (resourceAxis)
      return failure();
    resourceAxis = axis;
  }
  if (!resourceAxis)
    return failure();
  std::optional<unsigned> coordinate;
  for (auto [position, axis] : llvm::enumerate(load.getSourceAxes())) {
    if (axis != *resourceAxis)
      continue;
    if (coordinate)
      return failure();
    coordinate = position;
  }
  return coordinate ? FailureOr<unsigned>(*coordinate)
                    : FailureOr<unsigned>(failure());
}

FailureOr<unsigned> PhysicalProgramAnalysis::coordinateIndex(
    ValueRange coordinates, PhysicalSourceAxis source) const {
  PhysicalAxisProjection result = queryCoordinateIndex(coordinates, source);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
}

} // namespace intent::gpu
