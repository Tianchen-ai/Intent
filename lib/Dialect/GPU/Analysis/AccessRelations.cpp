#include "AccessRelations.h"
#include "IndexBounds.h"
#include "ScalarExpressions.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu {
using namespace detail;

namespace {

bool sameBroadcastCoordinateExpression(Value lhs, Value rhs) {
  auto leftType = dyn_cast<FragmentType>(lhs.getType());
  auto rightType = dyn_cast<FragmentType>(rhs.getType());
  if (!leftType || !rightType)
    return false;
  FragmentType common = leftType.getShape().size() > rightType.getShape().size()
                            ? leftType : rightType;
  auto project = [&](FragmentType source) {
    auto projection = queryBroadcastProjection(source, common);
    if (projection.isExact())
      return projection;
    auto permutation = queryAxisPermutation(source, common);
    if (!permutation)
      return projection;
    for (auto [targetAxis, sourceAxis] : llvm::enumerate(*permutation))
      if (source.getShape()[sourceAxis] != common.getShape()[targetAxis] &&
          constantPhysicalExpression(
              cast<PhysicalExprAttr>(source.getShape()[sourceAxis])) != 1)
        return projection;
    projection.targetToSource.clear();
    for (int64_t sourceAxis : *permutation)
      projection.targetToSource.push_back(sourceAxis);
    projection.state = BroadcastProjectionState::Exact;
    return projection;
  };
  auto leftProjection = project(leftType);
  auto rightProjection = project(rightType);
  if (!leftProjection.isExact() || !rightProjection.isExact())
    return false;
  using Axes = SmallVector<std::optional<unsigned>, 4>;
  auto element = [](Type type) {
    auto fragment = dyn_cast<FragmentType>(type);
    return fragment ? fragment.getElementType() : type;
  };
  auto projectOperand = [&](Value operand, Value parent,
                            ArrayRef<std::optional<unsigned>> axes)
      -> FailureOr<Axes> {
    Axes projected(axes.size());
    auto source = dyn_cast<FragmentType>(operand.getType());
    if (!source)
      return projected;
    auto target = dyn_cast<FragmentType>(parent.getType());
    if (!target)
      return failure();
    auto projection = queryBroadcastProjection(source, target);
    if (!projection.isExact())
      return failure();
    for (auto [axis, parentAxis] : llvm::enumerate(axes))
      if (parentAxis)
        projected[axis] = projection.targetToSource[*parentAxis];
    return projected;
  };
  auto sameLanes = [](Type type, ArrayRef<std::optional<unsigned>> left,
                      ArrayRef<std::optional<unsigned>> right) {
    auto fragment = dyn_cast<FragmentType>(type);
    if (!fragment)
      return true;
    auto varying = [&](std::optional<unsigned> axis) {
      if (axis) {
        auto extent = cast<PhysicalExprAttr>(fragment.getShape()[*axis]);
        if (extent.getKind() ==
                PhysicalExprKind::Constant &&
            extent.getValue() == 1)
          return std::optional<unsigned>();
      }
      return axis;
    };
    return llvm::all_of(llvm::zip(left, right), [&](auto pair) {
      return varying(std::get<0>(pair)) == varying(std::get<1>(pair));
    });
  };
  std::function<bool(Value, Axes, Value, Axes, unsigned)> equivalent;
  equivalent = [&](Value left, Axes leftAxes, Value right, Axes rightAxes,
                   unsigned depth) -> bool {
    if (depth >= 32 || element(left.getType()) != element(right.getType()))
      return false;
    if (left == right)
      return sameLanes(left.getType(), leftAxes, rightAxes);
    auto unwrap = [&](Value current, Axes axes)
        -> std::optional<std::pair<Value, Axes>> {
      Value input;
      if (auto broadcast = current.getDefiningOp<BroadcastOp>())
        input = broadcast.getValue();
      else if (auto splat = current.getDefiningOp<SplatOp>())
        input = splat.getValue();
      else if (auto transpose = current.getDefiningOp<TransposeOp>()) {
        auto relations = queryFragmentOperandRelations(transpose.getOperation());
        if (failed(relations))
          return std::nullopt;
        for (std::optional<unsigned> &axis : axes)
          if (axis)
            axis = relations->front().correspondingSourceAxis(*axis);
        return std::make_pair(transpose.getValue(), std::move(axes));
      } else if (auto reshape = current.getDefiningOp<ReshapeOp>()) {
        auto source = cast<FragmentType>(reshape.getValue().getType());
        auto target = cast<FragmentType>(current.getType());
        if (source.getShape() == target.getShape())
          return std::make_pair(reshape.getValue(), std::move(axes));
        auto relations = queryFragmentOperandRelations(reshape.getOperation());
        if (failed(relations)) return std::nullopt;
        Axes resultToSource(target.getShape().size());
        for (const FragmentAxisGroup &group : relations->front().groups) {
          if (group.sourceAxes.empty() || group.resultAxes.empty())
            continue;
          if (group.sourceAxes.size() != 1 || group.resultAxes.size() != 1)
            return std::nullopt;
          unsigned sourceAxis = group.sourceAxes[0];
          unsigned resultAxis = group.resultAxes[0];
          if (source.getShape()[sourceAxis] != target.getShape()[resultAxis])
            return std::nullopt;
          resultToSource[resultAxis] = sourceAxis;
        }
        Axes projected(axes.size());
        for (auto [axis, resultAxis] : llvm::enumerate(axes))
          if (resultAxis)
            projected[axis] = resultToSource[*resultAxis];
        return std::make_pair(reshape.getValue(), std::move(projected));
      }
      if (!input)
        return std::nullopt;
      FailureOr<Axes> projected = projectOperand(input, current, axes);
      return succeeded(projected)
                 ? std::optional(std::make_pair(input, std::move(*projected)))
                 : std::nullopt;
    };
    if (auto unwrapped = unwrap(left, leftAxes))
      return equivalent(unwrapped->first, std::move(unwrapped->second), right,
                        std::move(rightAxes), depth + 1);
    if (auto unwrapped = unwrap(right, rightAxes))
      return equivalent(left, std::move(leftAxes), unwrapped->first,
                        std::move(unwrapped->second), depth + 1);
    if (!isa<FragmentType>(left.getType()) &&
        !isa<FragmentType>(right.getType()))
      return sameScalarExpression(left, right);
    auto leftRange = left.getDefiningOp<MakeRangeOp>();
    auto rightRange = right.getDefiningOp<MakeRangeOp>();
    if (leftRange || rightRange)
      return leftRange && rightRange &&
             sameLanes(left.getType(), leftAxes, rightAxes) &&
             sourceAxisIdentity(leftRange) == sourceAxisIdentity(rightRange) &&
             samePhysicalScalarExpression(leftRange.getStart(), rightRange.getStart()) &&
             samePhysicalScalarExpression(leftRange.getExtent(), rightRange.getExtent()) &&
             samePhysicalScalarExpression(leftRange.getStep(), rightRange.getStep());
    auto leftBinary = left.getDefiningOp<BinaryOp>();
    auto rightBinary = right.getDefiningOp<BinaryOp>();
    if (!leftBinary || !rightBinary ||
        leftBinary.getOperatorKind() != rightBinary.getOperatorKind() ||
        leftBinary.getApproximate() != rightBinary.getApproximate() ||
        leftBinary.getFlushToZero() != rightBinary.getFlushToZero() ||
        leftBinary.getStrictRounding() != rightBinary.getStrictRounding())
      return false;
    for (auto [leftOperand, rightOperand] : llvm::zip(leftBinary->getOperands(),
                                                    rightBinary->getOperands())) {
      FailureOr<Axes> leftMapping = projectOperand(leftOperand, left, leftAxes);
      FailureOr<Axes> rightMapping = projectOperand(rightOperand, right, rightAxes);
      if (failed(leftMapping) || failed(rightMapping) ||
          !equivalent(leftOperand, *leftMapping, rightOperand, *rightMapping,
                      depth + 1))
        return false;
    }
    return true;
  };
  return equivalent(lhs, leftProjection.targetToSource, rhs,
                    rightProjection.targetToSource, 0);
}

bool linearizedGatherWithinResource(Value coordinate, Value resource) {
  auto sourceReshape = resource.getDefiningOp<ReshapeOp>();
  auto source = sourceReshape
                    ? dyn_cast<FragmentType>(sourceReshape.getValue().getType())
                    : FragmentType();
  auto flatSource = dyn_cast<FragmentType>(resource.getType());
  if (!source || !flatSource || source.getShape().size() < 2 ||
      flatSource.getShape().size() != 1)
    return false;
  auto binary = [](Value value, BinaryOperator kind) {
    auto operation = stripBroadcast(value).getDefiningOp<BinaryOp>();
    Type type = value.getType();
    if (auto fragment = dyn_cast<FragmentType>(type))
      type = fragment.getElementType();
    return operation && operation.getOperatorKind() == kind && type.isIndex()
               ? operation : BinaryOp();
  };
  auto offset = binary(coordinate, BinaryOperator::Add);
  auto tail = offset ? binary(offset.getRhs(), BinaryOperator::Remainder)
                     : BinaryOp();
  auto prefix = offset ? binary(offset.getLhs(), BinaryOperator::Add)
                       : BinaryOp();
  auto outer = prefix ? binary(prefix.getLhs(), BinaryOperator::Multiply)
                      : BinaryOp();
  auto selected = prefix ? binary(prefix.getRhs(), BinaryOperator::Multiply)
                         : BinaryOp();
  auto quotient = outer ? binary(outer.getLhs(), BinaryOperator::FloorDivide)
                        : BinaryOp();
  auto sourceStride = outer ? binary(outer.getRhs(), BinaryOperator::Multiply)
                            : BinaryOp();
  auto resultStride = quotient
                          ? binary(quotient.getRhs(), BinaryOperator::Multiply)
                          : BinaryOp();
  if (!tail || !selected || !sourceStride || !resultStride ||
      !sameScalarExpression(quotient.getLhs(), tail.getLhs()) ||
      !sameScalarExpression(selected.getRhs(), tail.getRhs()) ||
      !sameScalarExpression(sourceStride.getRhs(), tail.getRhs()) ||
      !sameScalarExpression(resultStride.getRhs(), tail.getRhs()))
    return false;
  auto ordinal = quotient.getLhs().getDefiningOp<MakeRangeOp>();
  Value index = stripBroadcast(selected.getLhs());
  uint64_t castLimit = std::numeric_limits<int64_t>::max();
  auto elementType = [](Type type) {
    if (auto fragment = dyn_cast<FragmentType>(type))
      return fragment.getElementType();
    return type;
  };
  while (auto cast = index.getDefiningOp<CastOp>()) {
    Type sourceType = elementType(cast.getValue().getType());
    Type resultType = elementType(cast.getResult().getType());
    if (!isa<IndexType, IntegerType>(sourceType) ||
        !isa<IndexType, IntegerType>(resultType))
      return false;
    if (auto integer = dyn_cast<IntegerType>(resultType);
        integer && integer.getWidth() < 64) {
      unsigned valueBits = integer.getWidth() - (integer.isUnsigned() ? 0 : 1);
      castLimit = std::min(castLimit, (uint64_t{1} << valueBits) - 1);
    }
    index = stripBroadcast(cast.getValue());
  }
  auto indexReshape = index.getDefiningOp<ReshapeOp>();
  auto indices = indexReshape
                     ? dyn_cast<FragmentType>(indexReshape.getValue().getType())
                     : FragmentType();
  auto flatIndices = dyn_cast<FragmentType>(index.getType());
  if (!ordinal || !indices || !flatIndices ||
      indices.getShape().size() != source.getShape().size() ||
      flatIndices.getShape().size() != 1 ||
      integerConstant(ordinal.getStart()) != 0 ||
      !isUnitStepValue(ordinal.getStep()) ||
      !matchesResourceExtent(ordinal.getExtent(), index, 0))
    return false;

  auto kernel = sourceReshape->getParentOfType<func::FuncOp>();
  auto positiveExtentLimit = [&](PhysicalExprAttr extent) -> std::optional<int64_t> {
    auto bounds = positiveExtentBounds(kernel, extent);
    return bounds ? std::optional<int64_t>(bounds->second) : std::nullopt;
  };
  if (!positiveExtentLimit(resourceExtentExpression(resource, 0)) ||
      !positiveExtentLimit(resourceExtentExpression(index, 0)))
    return false;

  for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
    if (!matchesResourceExtent(sourceStride.getLhs(), sourceReshape.getValue(), axis) ||
        !matchesResourceExtent(resultStride.getLhs(), indexReshape.getValue(), axis))
      continue;
    bool sameOtherAxes = true;
    for (unsigned other = 0; other < source.getShape().size(); ++other)
      sameOtherAxes &= other == axis ||
                       (source.getShape()[other] == indices.getShape()[other] &&
                        source.getAxisMaps()[other] == indices.getAxisMaps()[other]);
    if (!sameOtherAxes)
      continue;
    auto selectedLimit = positiveExtentLimit(
        cast<PhysicalExprAttr>(source.getShape()[axis]));
    if (!selectedLimit || static_cast<uint64_t>(*selectedLimit - 1) > castLimit)
      continue;
    MLIRContext *context = resource.getContext();
    auto inner = PhysicalExprAttr::get(
        context, PhysicalExprKind::Constant, 1,
        StringAttr::get(context, ""), ArrayAttr::get(context, {}));
    for (unsigned dimension = axis + 1; dimension < source.getShape().size(); ++dimension) {
      auto extent = cast<PhysicalExprAttr>(source.getShape()[dimension]);
      inner = dimension == axis + 1
                  ? extent
                  : PhysicalExprAttr::get(
                        context, PhysicalExprKind::Multiply,
                        0, StringAttr::get(context, ""),
                        ArrayAttr::get(context, {inner, extent}));
    }
    if (!valueMatchesExtent(stripBroadcast(tail.getRhs()), inner) ||
        !coordinateRangeWithinResource(indexReshape.getValue(),
                                       sourceReshape.getValue(), axis))
      continue;
    // The exact reshapes preserve row-major element order.  For a bounded
    // ordinal, q < outer and remainder < inner; the selected index is < B.
    // Thus q*(B*inner) + index*inner + remainder is in [0, outer*B*inner).
    // The positive finite shape limits above also exclude index overflow.
    return true;
  }
  return false;
}

} // namespace

namespace detail {

bool matchesResourceExtent(Value value, Value resource, unsigned axis) {
  PhysicalExprAttr extent = resourceExtentExpression(resource, axis);
  return extent && valueMatchesExtent(stripScalarIdentity(value), extent);
}

bool capacityCoversResourceExtent(Value value, Value resource, unsigned axis) {
  value = stripScalarIdentity(value);
  ParameterAttr parameter = queryParameter(value);
  if (auto expression = value.getDefiningOp<PhysicalExprOp>();
      expression && expression.getExpression().getKind() ==
                        PhysicalExprKind::Parameter) {
    auto resolved = queryParameterBySymbol(
        expression->getParentOfType<func::FuncOp>(),
        expression.getExpression().getParameterReference().getName());
    if (succeeded(resolved))
      parameter = *resolved;
  }
  if (!parameter || parameter.getCategory() !=
                        ParameterCategory::Coverage)
    return false;
  auto covered = parameter.getBinding().getCoverageBound();
  return covered && covered == resourceExtentExpression(resource, axis);
}

bool upperBoundWithinResource(Value value, Value resource, unsigned axis) {
  if (matchesResourceExtent(value, resource, axis))
    return true;
  PhysicalExprAttr extent = resourceExtentExpression(resource, axis);
  return extent && IndexRelations().atMost(stripScalarIdentity(value), extent);
}

bool derivesFromAccessCoordinate(Value value, Value coordinate) {
  // Coordinate replay can duplicate a pure expression before CSE. Its bounds
  // still apply when the complete typed expression and SSA leaves are equal.
  if (sameScalarExpression(value, coordinate))
    return true;
  auto valueType = dyn_cast<FragmentType>(value.getType());
  auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
  if (valueType && coordinateType &&
      queryBroadcastProjection(valueType, coordinateType).isExact() &&
      sameScalarExpression(stripBroadcast(value), stripBroadcast(coordinate)))
    return true;
  if (sameBroadcastCoordinateExpression(value, coordinate))
    return true;
  value = stripIntegerIndexCasts(value);
  coordinate = stripIntegerIndexCasts(coordinate);
  if (sameScalarExpression(value, coordinate))
    return true;
  auto valueRange = value.getDefiningOp<MakeRangeOp>();
  auto coordinateRange = coordinate.getDefiningOp<MakeRangeOp>();
  return valueRange && coordinateRange &&
         sourceAxisIdentity(valueRange) == sourceAxisIdentity(coordinateRange) &&
         sameScalarExpression(valueRange.getStart(),
                              coordinateRange.getStart()) &&
         sameScalarExpression(valueRange.getExtent(),
                              coordinateRange.getExtent()) &&
         sameScalarExpression(valueRange.getStep(), coordinateRange.getStep());
}

bool isInclusiveCoordinateUpperBound(Value value, Value coordinate) {
  coordinate = stripBroadcast(coordinate);
  if (!value.getType().isIndex() || !coordinate.getType().isIndex())
    return false;
  PhysicalExprAttr upper = queryNonNegativeIndexUpperBound(coordinate);
  return upper && queryLaunchExpression(value) == upper;
}

bool coordinateRangeWithinResource(Value coordinate, Value resource,
                                   unsigned axis) {
  if (axis == 0 && linearizedGatherWithinResource(coordinate, resource))
    return true;
  if (IndexRelations().coordinateInBounds(
          coordinate, resourceExtentExpression(resource, axis))) return true;
  coordinate = stripIntegerIndexCasts(coordinate);
  // Reassociation and permutation preserve the set of coordinate values.
  if (auto reshape = coordinate.getDefiningOp<ReshapeOp>())
    return coordinateRangeWithinResource(reshape.getValue(), resource, axis);
  if (auto transpose = coordinate.getDefiningOp<TransposeOp>())
    return coordinateRangeWithinResource(transpose.getValue(), resource, axis);
  if (auto select = coordinate.getDefiningOp<SelectOp>()) {
    if (!coordinateRangeWithinResource(select.getFalseValue(), resource, axis))
      return false;
    if (coordinateRangeWithinResource(select.getTrueValue(), resource, axis))
      return true;
    bool lower = false;
    bool upper = false;
    std::function<void(Value)> inspectPredicate = [&](Value predicate) {
      predicate = stripBroadcast(predicate);
      if (auto reshape = predicate.getDefiningOp<ReshapeOp>()) {
        auto source = cast<FragmentType>(reshape.getValue().getType());
        auto target = cast<FragmentType>(predicate.getType());
        auto unit = [](Attribute attribute) {
          auto extent = cast<PhysicalExprAttr>(attribute);
          return extent.getKind() ==
                     PhysicalExprKind::Constant &&
                 extent.getValue() == 1;
        };
        auto relations = queryFragmentOperandRelations(reshape.getOperation());
        bool projection = succeeded(relations) && llvm::all_of(
            relations->front().groups, [&](const FragmentAxisGroup &group) {
              ArrayRef<unsigned> inputs = group.sourceAxes;
              ArrayRef<unsigned> outputs = group.resultAxes;
              if (inputs.empty())
                return llvm::all_of(outputs, [&](unsigned axis) {
                  return unit(target.getShape()[axis]);
                });
              if (outputs.empty())
                return llvm::all_of(inputs, [&](unsigned axis) {
                  return unit(source.getShape()[axis]);
                });
              return inputs.size() == 1 && outputs.size() == 1 &&
                     source.getShape()[inputs[0]] == target.getShape()[outputs[0]];
            });
        if (projection)
          inspectPredicate(reshape.getValue());
        return;
      }
      if (integerConstant(predicate) == 0) {
        lower = upper = true;
        return;
      }
      if (auto conjunction = predicate.getDefiningOp<BinaryOp>()) {
        if (conjunction.getOperatorKind() == BinaryOperator::LogicalAnd ||
            conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd) {
          inspectPredicate(conjunction.getLhs());
          inspectPredicate(conjunction.getRhs());
        }
        return;
      }
      auto compare = predicate.getDefiningOp<CompareOp>();
      if (!compare)
        return;
      Value selected = select.getTrueValue();
      bool lhsIndex = derivesFromAccessCoordinate(compare.getLhs(), selected);
      bool rhsIndex = derivesFromAccessCoordinate(compare.getRhs(), selected);
      if ((compare.getPredicate() == ComparePredicate::Ge && lhsIndex &&
           integerConstant(compare.getRhs()) == 0) ||
          (compare.getPredicate() == ComparePredicate::Le && rhsIndex &&
           integerConstant(compare.getLhs()) == 0))
        lower = true;
      if ((compare.getPredicate() == ComparePredicate::Lt && lhsIndex &&
           upperBoundWithinResource(compare.getRhs(), resource, axis)) ||
          (compare.getPredicate() == ComparePredicate::Gt && rhsIndex &&
           upperBoundWithinResource(compare.getLhs(), resource, axis)))
        upper = true;
    };
    inspectPredicate(select.getCondition());
    return lower && upper;
  }
  return false;
}

bool hasExactPhysicalRangeCoverage(Value coordinate, Value upperBound) {
  return IndexRelations().hasExactRangeEndpoint(coordinate, upperBound);
}

} // namespace detail

} // namespace intent::gpu
