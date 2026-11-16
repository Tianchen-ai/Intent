#include "TilePlan.h"
#include "Bounds.h"
#include "Coordinates.h"
#include "Intent/Dialect/Block/Analysis/IndexRelations.h"
#include "Intent/Dialect/Block/Analysis/ProgramInterface.h"
#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "Intent/Dialect/Block/IR/PhysicalExpressions.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::cutile {

namespace {

bool isUnitExtent(Attribute attribute) {
  auto extent = dyn_cast<block::PhysicalExprAttr>(attribute);
  return extent &&
         extent.getKind() ==
             block::PhysicalExprKind::Constant &&
         extent.getValue() == 1;
}

bool collectTileCoordinates(Value value, NativeTileAxisPlan &axis,
                            int64_t scale = 1) {
  if (auto range = value.getDefiningOp<block::MakeRangeOp>()) {
    if (llvm::any_of(axis.ranges, [&](auto term) { return term.first == range; }))
      return false;
    axis.ranges.emplace_back(range, scale);
    return true;
  }
  if (Value scalar = uniformScalarFill(value)) {
    if (!scalar.getType().isIndex())
      return false;
    axis.offsets.emplace_back(scalar, scale);
    return true;
  }
  if (auto broadcast = value.getDefiningOp<block::BroadcastOp>())
    return collectTileCoordinates(broadcast.getValue(), axis, scale);
  if (auto reshape = value.getDefiningOp<block::ReshapeOp>())
    return collectTileCoordinates(reshape.getValue(), axis, scale);
  if (auto transpose = value.getDefiningOp<block::TransposeOp>())
    return collectTileCoordinates(transpose.getValue(), axis, scale);
  auto binary = value.getDefiningOp<block::BinaryOp>();
  if (!binary)
    return false;
  if (binary.getOperatorKind() == BinaryOperator::Add)
    return collectTileCoordinates(binary.getLhs(), axis, scale) &&
           collectTileCoordinates(binary.getRhs(), axis, scale);
  if (binary.getOperatorKind() == BinaryOperator::Subtract) {
    // Translate a contiguous tile without admitting a reversed varying axis.
    Value offset = uniformScalarFill(binary.getRhs());
    int64_t negativeScale;
    if (!offset || !offset.getType().isIndex() ||
        llvm::MulOverflow(scale, int64_t{-1}, negativeScale) ||
        !collectTileCoordinates(binary.getLhs(), axis, scale)) return false;
    axis.offsets.emplace_back(offset, negativeScale);
    return true;
  }
  if (binary.getOperatorKind() == BinaryOperator::Multiply) {
    for (unsigned i = 0; i != 2; ++i) {
      Value scalar = uniformScalarFill(binary->getOperand(i));
      auto factor = scalar ? block::IndexRelations().constant(scalar) : std::nullopt;
      int64_t product;
      if (factor && *factor > 0 && !llvm::MulOverflow(scale, *factor, product))
        return collectTileCoordinates(binary->getOperand(1 - i), axis, product);
    }
  }
  return false;
}

} // namespace

FailureOr<NativeTileAccessPlan> analyzeNativeTileAccess(
    block::AccessOpInterface access, func::FuncOp kernel,
    const block::PhysicalAccessBoundsFact &accessBounds) {
  Operation *owner = access.getOperation();
  auto view = dyn_cast<block::ViewType>(access.getAccessResource().getType());
  auto computationType = dyn_cast<block::FragmentType>(access.getAccessValueType());
  if (!view || !computationType) return failure();
  auto coordinates = access.getAccessCoordinates();
  auto sourceAxes = access.getAccessSourceAxes();
  const unsigned resourceRank = view.getRank();
  const unsigned computationRank = computationType.getShape().size();
  if (coordinates.size() != resourceRank ||
      sourceAxes.size() != resourceRank || computationRank == 0)
    return failure();

  SmallVector<Value> resourceCoordinates(resourceRank);
  SmallVector<unsigned> coordinateSlots(resourceRank);
  for (auto [slot, coordinate] : llvm::enumerate(coordinates)) {
    int64_t sourceAxis = sourceAxes[slot];
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(resourceRank) ||
        resourceCoordinates[sourceAxis])
      return failure();
    resourceCoordinates[sourceAxis] = coordinate;
    coordinateSlots[sourceAxis] = slot;
  }

  NativeTileAccessPlan plan;
  plan.axes.resize(resourceRank);
  plan.toComputation.assign(computationRank, -1);
  plan.toResource.assign(computationRank, -1);
  SmallVector<bool> usedComputationAxis(computationRank, false);
  SmallVector<unsigned> unresolvedScalarAxes;
  ArrayRef<int64_t> dimensions =
      view.getLayout().getDimensionIds().asArrayRef();
  if (dimensions.size() != resourceRank)
    return failure();

  auto assignComputationAxis = [&](unsigned resourceAxis,
                                   unsigned computationAxis) {
    if (computationAxis >= computationRank ||
        usedComputationAxis[computationAxis])
      return false;
    usedComputationAxis[computationAxis] = true;
    plan.axes[resourceAxis].computationAxes.push_back(computationAxis);
    return true;
  };

  for (unsigned resourceAxis = 0; resourceAxis < resourceRank;
       ++resourceAxis) {
    NativeTileAxisPlan &axis = plan.axes[resourceAxis];
    Value coordinate = resourceCoordinates[resourceAxis];
    Value scalar = uniformScalarFill(coordinate);
    if (!scalar) {
      Value expression = coordinate;
      while (isa_and_nonnull<block::BroadcastOp, block::ReshapeOp, block::TransposeOp>(
          expression.getDefiningOp()))
        expression = expression.getDefiningOp()->getOperand(0);
      if (auto binary = expression.getDefiningOp<block::BinaryOp>()) {
        Value rhs = uniformScalarFill(binary.getRhs());
        auto constant = rhs ? block::IndexRelations().constant(rhs) : std::nullopt;
        auto kind = binary.getOperatorKind();
        int64_t divisor = 0;
        if (kind == BinaryOperator::FloorDivide && constant && *constant > 0)
          divisor = *constant;
        if (kind == BinaryOperator::RightShift && constant &&
            *constant >= 0 && *constant < 63)
          divisor = int64_t{1} << *constant;
        if (divisor) {
          if (!collectTileCoordinates(binary.getLhs(), axis) ||
              axis.ranges.size() != 1 || axis.ranges.front().second != 1 ||
              !axis.offsets.empty() || !block::isUnitStepRange(axis.ranges.front().first))
            return failure();
          // The quotient is a unit resource axis only under the per-tile
          // uniformity guard emitted with its native indices.
          axis.scalarIndex = axis.ranges.front().first.getStart();
          axis.divisor = divisor;
          axis.originInBounds =
              llvm::is_contained(accessBounds.assumedAxes, resourceAxis);
          auto loop = completeAlignedTileLoop(
              axis.scalarIndex, axis.ranges.front().first.getExtent());
          auto upper = loop ? block::queryLaunchExpression(loop.getUpperBound())
                            : block::PhysicalExprAttr();
          if (upper && upper.getKind() ==
                           block::PhysicalExprKind::Multiply &&
              upper.getOperands().size() == 2) {
            auto covers = [&](Attribute dimension, Attribute factor) {
              auto dim = cast<block::PhysicalExprAttr>(dimension);
              auto scale = cast<block::PhysicalExprAttr>(factor);
              return dim.getKind() ==
                         block::PhysicalExprKind::Dimension &&
                     dim.getValue() == dimensions[resourceAxis] &&
                     scale.getKind() ==
                         block::PhysicalExprKind::Constant &&
                     scale.getValue() == divisor;
            };
            axis.originInBounds |= covers(upper.getOperands()[0], upper.getOperands()[1]) ||
                                   covers(upper.getOperands()[1], upper.getOperands()[0]);
          }
          continue;
        }
        if (kind == BinaryOperator::Remainder) {
          if (!rhs || !rhs.getType().isIndex())
            return failure();
          axis.modulus = rhs;
          coordinate = binary.getLhs();
        } else if (kind == BinaryOperator::BitwiseAnd) {
          if (!constant)
            return failure();
          const int64_t mask = constant.value();
          int64_t period;
          if (mask < 0 || llvm::AddOverflow(mask, int64_t{1}, period) ||
              !llvm::isPowerOf2_64(period))
            return failure();
          axis.modulus = IntegerAttr::get(IndexType::get(owner->getContext()), period);
          coordinate = binary.getLhs();
        }
      }
      if (!collectTileCoordinates(coordinate, axis) || axis.ranges.empty())
        return failure();
      llvm::stable_sort(axis.ranges, [](auto lhs, auto rhs) {
        return lhs.second > rhs.second;
      });
      for (auto [range, stride] : axis.ranges) {
        auto resultAxis = nativeAccessRangeAxis(access, coordinateSlots[resourceAxis], range);
        auto rangeType = dyn_cast<block::FragmentType>(range.getResult().getType());
        if (failed(resultAxis) || !assignComputationAxis(resourceAxis, *resultAxis) ||
            !block::isUnitStepRange(range) || !rangeType ||
            rangeType.getShape().size() != 1 ||
            rangeType.getShape()[0] !=
                computationType.getShape()[axis.computationAxes.back()])
          return failure();
      }
      axis.originInBounds =
          llvm::is_contained(accessBounds.assumedAxes, resourceAxis);
      if (axis.modulus && axis.ranges.size() == 1 &&
          axis.ranges.front().second == 1 && axis.offsets.empty()) {
        auto range = axis.ranges.front().first;
        std::optional<int64_t> modulus;
        if (auto value = dyn_cast<Value>(axis.modulus))
          modulus = block::IndexRelations().constant(value);
        else
          modulus = cast<IntegerAttr>(cast<Attribute>(axis.modulus)).getInt();
        if (modulus && *modulus > 0 &&
            completeAlignedTileLoop(range.getStart(), range.getExtent()))
          axis.originInBounds |= constantOriginInView(
              *modulus - 1, view, resourceAxis, kernel);
      }
      if (!axis.modulus && axis.ranges.size() == 1 &&
          axis.ranges.front().second == 1 &&
          llvm::all_of(axis.offsets, [](auto term) { return term.second == 1; })) {
        SmallVector<Value> offsets;
        for (auto term : axis.offsets)
          offsets.push_back(term.first);
        axis.originInBounds |= rangeOriginInView(
            axis.ranges.front().first, offsets, view, resourceAxis,
            dimensions[resourceAxis], kernel);
      }
      continue;
    }

    if (!scalar || !scalar.getType().isIndex())
      return failure();
    axis.scalarIndex = scalar;
    axis.originInBounds =
        scalarOriginInView(scalar, view, resourceAxis,
                           dimensions[resourceAxis], kernel) ||
        llvm::is_contained(accessBounds.assumedAxes, resourceAxis);
    Value strippedScalar = stripIndexIdentities(scalar);
    if (auto workset =
            strippedScalar.getDefiningOp<block::WorksetCoordinateOp>()) {
      block::PhysicalDimensionProjection projection =
          block::queryFragmentDimension(computationType,
                                      workset.getDimensionId());
      if (projection.state == block::PhysicalFactState::Ambiguous)
        return failure();
      if (projection.isExact()) {
        if (!assignComputationAxis(resourceAxis, projection.fragmentAxis) ||
            !isUnitExtent(
                computationType.getShape()[axis.computationAxes.back()]))
          return failure();
      }
    } else {
      unresolvedScalarAxes.push_back(resourceAxis);
    }
  }

  SmallVector<unsigned> remainingUnitAxes;
  for (unsigned computationAxis = 0; computationAxis < computationRank;
       ++computationAxis)
    if (!usedComputationAxis[computationAxis] &&
        isUnitExtent(computationType.getShape()[computationAxis]))
      remainingUnitAxes.push_back(computationAxis);
  if (unresolvedScalarAxes.size() == 1 && remainingUnitAxes.size() == 1) {
    unsigned resourceAxis = unresolvedScalarAxes.front();
    unsigned computationAxis = remainingUnitAxes.front();
    if (!assignComputationAxis(resourceAxis, computationAxis))
      return failure();
    NativeTileAxisPlan &axis = plan.axes[resourceAxis];
    axis.originInBounds =
        scalarOriginInView(axis.scalarIndex, view, resourceAxis,
                           dimensions[resourceAxis], kernel) ||
        llvm::is_contained(accessBounds.assumedAxes, resourceAxis);
  }

  if (llvm::any_of(usedComputationAxis, [](bool used) { return !used; }))
    return failure();

  MLIRContext *context = owner->getContext();
  auto unit = block::PhysicalExprAttr::get(
      context, block::PhysicalExprKind::Constant, 1,
      StringAttr::get(context), ArrayAttr::get(context, {}));
  SmallVector<Attribute> resourceShape(resourceRank);
  SmallVector<Attribute> resourceMappings(resourceRank);
  SmallVector<Attribute> packedShape;
  SmallVector<Attribute> packedMappings;
  packedShape.reserve(computationRank);
  packedMappings.reserve(computationRank);
  for (unsigned resourceAxis = 0; resourceAxis < resourceRank;
       ++resourceAxis) {
    ArrayRef<unsigned> computationAxes = plan.axes[resourceAxis].computationAxes;
    if (computationAxes.empty()) {
      resourceShape[resourceAxis] = unit;
      resourceMappings[resourceAxis] = block::AxisMapAttr::get(
          context, view.getSourceId(), resourceAxis,
          dimensions[resourceAxis], resourceAxis, false);
      continue;
    }
    Attribute extent = unit;
    auto mapping = cast<block::AxisMapAttr>(
        computationType.getAxisMaps()[computationAxes.front()]);
    for (unsigned computationAxis : computationAxes) {
      Attribute component = computationType.getShape()[computationAxis];
      if (isUnitExtent(extent))
        extent = component;
      else if (!isUnitExtent(component))
        extent = block::PhysicalExprAttr::get(
            context, block::PhysicalExprKind::Multiply, 0,
            StringAttr::get(context), ArrayAttr::get(context, {extent, component}));
      auto componentMap = cast<block::AxisMapAttr>(
          computationType.getAxisMaps()[computationAxis]);
      unsigned packedAxis = packedShape.size();
      packedShape.push_back(component);
      packedMappings.push_back(block::AxisMapAttr::get(
          context, componentMap.getSourceId(), componentMap.getSourceAxis(),
          componentMap.getDimensionId(), packedAxis, componentMap.getDerived()));
      plan.toComputation[computationAxis] = packedAxis;
      plan.toResource[packedAxis] = computationAxis;
    }
    resourceShape[resourceAxis] = extent;
    resourceMappings[resourceAxis] = block::AxisMapAttr::get(
        context, mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), resourceAxis,
        mapping.getDerived() || computationAxes.size() > 1);
  }

  plan.resourceType = block::FragmentType::get(
      context, computationType.getElementType(),
      ArrayAttr::get(context, resourceShape),
      ArrayAttr::get(context, resourceMappings), computationType.getValidity(),
      computationType.getOwner());
  plan.packedType = block::FragmentType::get(
      context, computationType.getElementType(),
      ArrayAttr::get(context, packedShape),
      ArrayAttr::get(context, packedMappings),
      computationType.getValidity(), computationType.getOwner());
  if (plan.resourceType != plan.packedType) {
    FailureOr<ArrayAttr> resourceToPacked =
        block::inferReshapeReassociation(plan.resourceType, plan.packedType);
    FailureOr<ArrayAttr> packedToResource =
        block::inferReshapeReassociation(plan.packedType, plan.resourceType);
    if (failed(resourceToPacked) || failed(packedToResource))
      return failure();
    plan.resourceToPacked = *resourceToPacked;
    plan.packedToResource = *packedToResource;
  }
  return plan;
}

} // namespace intent::cutile
