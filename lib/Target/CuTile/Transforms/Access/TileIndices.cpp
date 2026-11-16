#include "TileIndices.h"
#include "Bounds.h"
#include "Intent/Dialect/Block/Analysis/IndexRelations.h"
#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/IR/FragmentOpInterface.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;

namespace intent::cutile {

FailureOr<Value> tileIndex(OpBuilder &builder, Location location, Value start,
                           Value extent) {
  std::optional<int64_t> startConstant = block::IndexRelations().constant(start);
  std::optional<int64_t> extentConstant = block::IndexRelations().constant(extent);
  if (startConstant && extentConstant && *startConstant >= 0 &&
      *extentConstant > 0 && *startConstant % *extentConstant == 0)
    return builder
        .create<arith::ConstantIndexOp>(location,
                                        *startConstant / *extentConstant)
        .getResult();
  if (isProvably(start, 0))
    return builder.create<arith::ConstantIndexOp>(location, 0).getResult();
  if (auto coordinate = start.getDefiningOp<block::WorksetCoordinateOp>())
    return tileIndex(builder, location, coordinate.getCoordinate(), extent);
  if (auto binary = start.getDefiningOp<block::BinaryOp>()) {
    if (binary.getOperatorKind() == BinaryOperator::Multiply) {
      if (block::samePhysicalScalarExpression(binary.getLhs(), extent))
        return binary.getRhs();
      if (block::samePhysicalScalarExpression(binary.getRhs(), extent))
        return binary.getLhs();
      if (isProvably(binary.getLhs(), 1))
        return tileIndex(builder, location, binary.getRhs(), extent);
      if (isProvably(binary.getRhs(), 1))
        return tileIndex(builder, location, binary.getLhs(), extent);
    }
    if (binary.getOperatorKind() == BinaryOperator::Add) {
      if (isProvably(binary.getLhs(), 0))
        return tileIndex(builder, location, binary.getRhs(), extent);
      if (isProvably(binary.getRhs(), 0))
        return tileIndex(builder, location, binary.getLhs(), extent);
    }
    if (binary.getOperatorKind() == BinaryOperator::Subtract) {
      if (isProvably(binary.getRhs(), 0))
        return tileIndex(builder, location, binary.getLhs(), extent);
      auto induction = dyn_cast<BlockArgument>(binary.getLhs());
      auto loop = induction
                      ? dyn_cast_or_null<scf::ForOp>(
                            induction.getOwner()->getParentOp())
                      : scf::ForOp();
      // An iteration's offset from its lower bound is tile aligned even when
      // the lower bound itself is not (for example a dynamic retained slice).
      if (loop && induction == loop.getInductionVar() &&
          block::samePhysicalScalarExpression(loop.getLowerBound(),
                                            binary.getRhs()) &&
          block::samePhysicalScalarExpression(loop.getStep(), extent))
        return Value(builder.create<block::BinaryOp>(
            location, builder.getIndexType(), start, extent,
            BinaryOperator::FloorDivide));
    }
  }
  auto argument = dyn_cast<BlockArgument>(start);
  auto loop =
      argument
          ? dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())
          : scf::ForOp();
  if (loop && argument == loop.getInductionVar() &&
      block::samePhysicalScalarExpression(loop.getStep(), extent) &&
      block::IndexRelations().multipleOf(loop.getLowerBound(), extent))
    return Value(builder.create<block::BinaryOp>(
        location, builder.getIndexType(), start, extent,
        BinaryOperator::FloorDivide));
  return failure();
}

namespace {

Value materializeTileOrigin(OpBuilder &builder, Location location,
                            const NativeTileAxisPlan &axis) {
  if (axis.scalarIndex) {
    if (!axis.divisor)
      return axis.scalarIndex;
    Value divisor = builder.create<arith::ConstantIndexOp>(location, axis.divisor);
    return builder.create<block::BinaryOp>(location, builder.getIndexType(),
        axis.scalarIndex, divisor, BinaryOperator::FloorDivide);
  }
  Value origin;
  auto add = [&](Value value, int64_t scale) {
    if (origin && scale == -1) {
      origin = builder.createOrFold<block::BinaryOp>(
          location, builder.getIndexType(), origin, value,
          BinaryOperator::Subtract);
      return;
    }
    if (scale != 1) {
      Value coefficient = builder.create<arith::ConstantIndexOp>(location, scale);
      value = builder.create<block::BinaryOp>(
          location, builder.getIndexType(), value, coefficient,
          BinaryOperator::Multiply);
    }
    origin = origin ? Value(builder.createOrFold<block::BinaryOp>(
                          location, builder.getIndexType(), origin, value,
                          BinaryOperator::Add))
                    : value;
  };
  for (auto [range, scale] : axis.ranges)
    add(range.getStart(), scale);
  for (auto [offset, scale] : axis.offsets)
    add(offset, scale);
  return origin;
}

Value materializeTileModulus(OpBuilder &builder, Location location,
                             const NativeTileAxisPlan &axis) {
  if (auto value = dyn_cast<Value>(axis.modulus))
    return value;
  return builder.create<arith::ConstantIndexOp>(
      location, cast<IntegerAttr>(cast<Attribute>(axis.modulus)).getInt());
}

Value materializeLoopOriginGuard(OpBuilder &builder, Location location,
                                 const NativeTileAxisPlan &axis, Value extent) {
  if (axis.scalarIndex || axis.modulus || axis.ranges.size() != 1 ||
      axis.ranges.front().second != 1)
    return {};
  auto range = axis.ranges.front().first;
  auto loop = completeAlignedTileLoop(range.getStart(), range.getExtent());
  if (!loop || llvm::any_of(axis.offsets, [&](auto term) {
        return !loop.isDefinedOutsideOfLoop(term.first);
      }))
    return {};
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value offset = zero;
  for (auto [value, scale] : axis.offsets) {
    if (scale != 1) {
      Value coefficient = builder.create<arith::ConstantIndexOp>(location, scale);
      value = builder.create<block::BinaryOp>(location, builder.getIndexType(),
                                            value, coefficient,
                                            BinaryOperator::Multiply);
    }
    offset = builder.create<block::BinaryOp>(location, builder.getIndexType(),
                                          offset, value, BinaryOperator::Add);
  }
  // All iterations satisfy 0 <= iv < upper. Under this uniform condition,
  // offset + iv is inside the view and cannot wrap. Clamp only the proof's
  // subtraction operand so that evaluating the guard is safe even for negative
  // offsets; the actual access and its original per-origin guard are unchanged.
  Value nonnegative = builder.create<block::CompareOp>(
      location, builder.getI1Type(), offset, zero, ComparePredicate::Ge);
  Value positiveOffset = builder.create<block::BinaryOp>(
      location, builder.getIndexType(), offset, zero, BinaryOperator::Maximum);
  Value remaining = builder.create<block::BinaryOp>(
      location, builder.getIndexType(), extent, positiveOffset,
      BinaryOperator::Subtract);
  Value contained = builder.create<block::CompareOp>(
      location, builder.getI1Type(), loop.getUpperBound(), remaining,
      ComparePredicate::Le);
  return builder.create<block::BinaryOp>(location, builder.getI1Type(),
                                       nonnegative, contained,
                                       BinaryOperator::LogicalAnd);
}

Value materializeScaledOriginGuard(OpBuilder &builder, Location location,
                                   Value origin, Value extent) {
  auto add = origin.getDefiningOp<block::BinaryOp>();
  if (!add || add.getOperatorKind() != BinaryOperator::Add) return {};
  for (unsigned scaledSide : {0u, 1u}) {
    auto multiply = add->getOperand(scaledSide).getDefiningOp<block::BinaryOp>();
    if (!multiply || multiply.getOperatorKind() != BinaryOperator::Multiply)
      continue;
    auto inner = block::queryNonNegativeIndexUpperBound(add->getOperand(1 - scaledSide));
    if (!inner || inner.getKind() != block::PhysicalExprKind::Constant) continue;
    for (unsigned factorSide : {0u, 1u}) {
      auto factor = block::IndexRelations().constant(multiply->getOperand(factorSide));
      auto outer = block::queryNonNegativeIndexUpperBound(
          multiply->getOperand(1 - factorSide));
      if (!factor || *factor <= 0 || !outer) continue;
      // For nonnegative x/y, x*p+y is in [0,E) without signed overflow
      // whenever upper(x) <= floor((E-1-upper(y))/p). The guard's subtraction
      // is representable even for E=0; it never evaluates an overflowing product.
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value offset = builder.create<arith::ConstantIndexOp>(location, inner.getValue());
      Value pitch = builder.create<arith::ConstantIndexOp>(location, *factor);
      Value remaining = builder.create<block::BinaryOp>(
          location, builder.getIndexType(), extent, one, BinaryOperator::Subtract);
      remaining = builder.create<block::BinaryOp>(
          location, builder.getIndexType(), remaining, offset, BinaryOperator::Subtract);
      Value limit = builder.create<block::BinaryOp>(
          location, builder.getIndexType(), remaining, pitch, BinaryOperator::FloorDivide);
      Value bound = builder.create<block::PhysicalExprOp>(location, builder.getIndexType(), outer);
      return builder.create<block::CompareOp>(
          location, builder.getI1Type(), bound, limit, ComparePredicate::Le);
    }
  }
  return {};
}

} // namespace

FailureOr<Value> materializeTileOriginGuard(
    OpBuilder &builder, Operation *owner, Value resource,
    NativeTileAccessPlan &plan,
    const block::PhysicalAccessBoundaryFact &boundary) {
  Value condition;
  Value zero;
  for (int64_t rawAxis : boundary.boundaryAxes) {
    if (rawAxis < 0 || rawAxis >= static_cast<int64_t>(plan.axes.size()))
      return failure();
    unsigned axis = static_cast<unsigned>(rawAxis);
    NativeTileAxisPlan &axisPlan = plan.axes[axis];
    if (axisPlan.originInBounds)
      continue;
    Value origin = materializeTileOrigin(builder, owner->getLoc(), axisPlan);
    if (!origin)
      return failure();
    if (!axisPlan.modulus) {
      auto view = cast<block::ViewType>(resource.getType());
      auto dimensions = view.getLayout().getDimensionIds().asArrayRef();
      if (scalarOriginInView(origin, view, axis, dimensions[axis],
                             owner->getParentOfType<func::FuncOp>()))
        continue;
    }
    if (axisPlan.modulus)
      origin = builder.create<block::BinaryOp>(
          owner->getLoc(), builder.getIndexType(), origin,
          materializeTileModulus(builder, owner->getLoc(), axisPlan),
          BinaryOperator::Remainder);
    if (!zero)
      zero = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 0);
    Value extent = builder.create<block::DimOp>(
        owner->getLoc(), builder.getIndexType(), resource, axis);
    Value nonNegative = builder.create<block::CompareOp>(
        owner->getLoc(), builder.getI1Type(), origin, zero,
        ComparePredicate::Ge);
    Value belowExtent = builder.create<block::CompareOp>(
        owner->getLoc(), builder.getI1Type(), origin, extent,
        ComparePredicate::Lt);
    Value axisCondition = builder.create<block::BinaryOp>(
        owner->getLoc(), builder.getI1Type(), nonNegative, belowExtent,
        BinaryOperator::LogicalAnd);
    if (axisPlan.modulus) {
      Value modulus = materializeTileModulus(builder, owner->getLoc(), axisPlan);
      Value positive = builder.create<block::CompareOp>(
          owner->getLoc(), builder.getI1Type(), modulus, zero,
          ComparePredicate::Gt);
      Value withinView = builder.create<block::CompareOp>(
          owner->getLoc(), builder.getI1Type(), modulus, extent,
          ComparePredicate::Le);
      // With a positive period, every Euclidean remainder is in [0, period).
      // A shape-specialized period can discharge the bound for the whole loop,
      // even when the original (unwrapped) coordinate depends on its IV.
      Value allOriginsInBounds = builder.create<block::BinaryOp>(
          owner->getLoc(), builder.getI1Type(), positive, withinView,
          BinaryOperator::LogicalAnd);
      axisCondition = builder.create<block::BinaryOp>(
          owner->getLoc(), builder.getI1Type(), allOriginsInBounds, axisCondition,
          BinaryOperator::LogicalOr);
    } else if (block::PhysicalExprAttr upper =
                   block::queryNonNegativeIndexUpperBound(origin)) {
      Value bound = builder.create<block::PhysicalExprOp>(
          owner->getLoc(), builder.getIndexType(), upper);
      Value allOriginsInBounds = builder.create<block::CompareOp>(
          owner->getLoc(), builder.getI1Type(), bound, extent,
          ComparePredicate::Lt);
      // Specialization can discharge the whole mapped domain. Otherwise the
      // original per-origin predicate still determines exactly the same access.
      axisCondition = builder.create<block::BinaryOp>(
          owner->getLoc(), builder.getI1Type(), allOriginsInBounds, axisCondition,
          BinaryOperator::LogicalOr);
    }
    if (Value uniform = materializeLoopOriginGuard(
            builder, owner->getLoc(), axisPlan, extent))
      axisCondition = builder.create<block::BinaryOp>(
          owner->getLoc(), builder.getI1Type(), uniform, axisCondition,
          BinaryOperator::LogicalOr);
    if (!axisPlan.modulus)
      if (Value uniform = materializeScaledOriginGuard(
              builder, owner->getLoc(), origin, extent))
        axisCondition = builder.create<block::BinaryOp>(
            owner->getLoc(), builder.getI1Type(), uniform, axisCondition,
            BinaryOperator::LogicalOr);
    condition = condition
                    ? Value(builder.create<block::BinaryOp>(
                          owner->getLoc(), builder.getI1Type(), condition,
                          axisCondition, BinaryOperator::LogicalAnd))
                    : axisCondition;
  }
  return condition;
}

Value materializeFullRangeGuard(
    OpBuilder &builder, Location location,
    const block::PhysicalAccessBoundaryFact &boundary) {
  Value guard;
  for (auto [range, upper] : boundary.rangeBounds) {
    if (auto loop = completeAlignedTileLoop(range.getStart(), range.getExtent()))
      if (block::samePhysicalScalarExpression(loop.getUpperBound(), upper))
        continue;
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value origin = range.getStart();
    Value nonnegativeOrigin = builder.create<block::CompareOp>(
        location, builder.getI1Type(), origin, zero, ComparePredicate::Ge);
    Value nonnegativeUpper = builder.create<block::CompareOp>(
        location, builder.getI1Type(), upper, zero, ComparePredicate::Ge);
    Value nonnegative = builder.create<block::BinaryOp>(
        location, builder.getI1Type(), nonnegativeOrigin, nonnegativeUpper,
        BinaryOperator::LogicalAnd);
    // With nonnegative endpoints, subtraction cannot overflow. A failed guard
    // keeps the original predicated gather, including a partial logical tile.
    Value remaining = builder.create<block::BinaryOp>(
        location, builder.getIndexType(), upper, origin,
        BinaryOperator::Subtract);
    Value fullRange = builder.create<block::CompareOp>(
        location, builder.getI1Type(), remaining, range.getExtent(),
        ComparePredicate::Ge);
    Value condition = builder.create<block::BinaryOp>(
        location, builder.getI1Type(), nonnegative, fullRange,
        BinaryOperator::LogicalAnd);
    guard = guard ? Value(builder.create<block::BinaryOp>(
                        location, builder.getI1Type(), guard, condition,
                        BinaryOperator::LogicalAnd))
                  : condition;
  }
  return guard;
}

FailureOr<MaterializedTileIndices>
materializeTileIndices(OpBuilder &builder, Operation *owner,
                       const NativeTileAccessPlan &plan,
                       bool allowDynamicAlignment) {
  MaterializedTileIndices result;
  result.values.reserve(plan.axes.size());
  auto require = [&](Value condition) {
    result.alignment = result.alignment
                           ? Value(builder.create<block::BinaryOp>(
                                 owner->getLoc(), builder.getI1Type(),
                                 result.alignment, condition,
                                 BinaryOperator::LogicalAnd))
                           : condition;
  };
  for (auto [resourceAxis, axis] : llvm::enumerate(plan.axes)) {
    if (axis.scalarIndex) {
      Value index = materializeTileOrigin(builder, owner->getLoc(), axis);
      if (axis.divisor) {
        auto range = axis.ranges.front().first;
        Value extent = range.getExtent();
        if (!block::IndexRelations().positive(extent))
          return failure();
        Value divisor = builder.create<arith::ConstantIndexOp>(owner->getLoc(), axis.divisor);
        if (isAlignedPeriodicTile(axis.scalarIndex, extent, divisor)) {
          result.values.push_back(index);
          continue;
        }
        Value one = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 1);
        Value offset = builder.create<block::BinaryOp>(owner->getLoc(),
            builder.getIndexType(), extent, one, BinaryOperator::Subtract);
        Value last = builder.create<block::BinaryOp>(owner->getLoc(),
            builder.getIndexType(), axis.scalarIndex, offset, BinaryOperator::Add);
        Value lastIndex = builder.create<block::BinaryOp>(owner->getLoc(),
            builder.getIndexType(), last, divisor, BinaryOperator::FloorDivide);
        // Positive extent and divisor make equal endpoint quotients sufficient;
        // signed wraparound would change the quotient's sign and fail the guard.
        require(builder.create<block::CompareOp>(owner->getLoc(), builder.getI1Type(),
                                               index, lastIndex, ComparePredicate::Eq));
      }
      result.values.push_back(index);
      continue;
    }
    Value start = materializeTileOrigin(builder, owner->getLoc(), axis);
    Value extent;
    if (axis.ranges.size() == 1 && axis.ranges.front().second == 1) {
      auto range = axis.ranges.front().first;
      extent = range.getExtent();
    } else {
      extent = builder.create<block::PhysicalExprOp>(
          owner->getLoc(), builder.getIndexType(),
          cast<block::PhysicalExprAttr>(plan.resourceType.getShape()[resourceAxis]));
      Value inner = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 1);
      Value one = inner;
      // A varying axis is contiguous only when its stride equals the product
      // of the inner tile extents. Unit axes contribute an origin, not a gap.
      for (auto [range, stride] : llvm::reverse(axis.ranges)) {
        Value coefficient = builder.create<arith::ConstantIndexOp>(
            owner->getLoc(), stride);
        Value unit = builder.create<block::CompareOp>(
            owner->getLoc(), builder.getI1Type(), range.getExtent(), one,
            ComparePredicate::Eq);
        Value contiguous = builder.create<block::CompareOp>(
            owner->getLoc(), builder.getI1Type(), coefficient, inner,
            ComparePredicate::Eq);
        require(builder.create<block::BinaryOp>(
            owner->getLoc(), builder.getI1Type(), unit, contiguous,
            BinaryOperator::LogicalOr));
        inner = builder.create<block::BinaryOp>(
            owner->getLoc(), builder.getIndexType(), inner, range.getExtent(),
            BinaryOperator::Multiply);
      }
    }
    Value uniformPeriodicAlignment;
    if (axis.modulus) {
      if (!block::IndexRelations().positive(extent))
        return failure();
      Location location = owner->getLoc();
      Value modulus = materializeTileModulus(builder, location, axis);
      if (isAlignedPeriodicTile(start, extent, modulus) &&
          completeAlignedTileLoop(start, extent)) {
        Value remainder = builder.create<block::BinaryOp>(
            location, builder.getIndexType(), start, modulus,
            BinaryOperator::Remainder);
        result.values.push_back(builder.create<block::BinaryOp>(
            location, builder.getIndexType(), remainder, extent,
            BinaryOperator::FloorDivide));
        continue;
      }
      Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value maximum = builder.create<arith::ConstantIndexOp>(
          location, std::numeric_limits<int64_t>::max());
      require(builder.create<block::CompareOp>(
          location, builder.getI1Type(), modulus, zero, ComparePredicate::Gt));
      if (completeAlignedTileLoop(start, extent)) {
        // Complete aligned iterations are nonnegative and cannot overflow
        // before their last lane. If the period is also tile-aligned, *every*
        // iteration fits one period; leave this as a specialization predicate.
        Value remainder = builder.create<block::BinaryOp>(
            location, builder.getIndexType(), modulus, extent,
            BinaryOperator::Remainder);
        uniformPeriodicAlignment = builder.create<block::CompareOp>(
            location, builder.getI1Type(), remainder, zero, ComparePredicate::Eq);
      } else {
        require(builder.create<block::CompareOp>(
            location, builder.getI1Type(), start, zero, ComparePredicate::Ge));
        Value available = builder.create<block::BinaryOp>(
            location, builder.getIndexType(), maximum, start,
            BinaryOperator::Subtract);
        Value lastOffset = builder.create<block::BinaryOp>(
            location, builder.getIndexType(), extent, one,
            BinaryOperator::Subtract);
        require(builder.create<block::CompareOp>(
            location, builder.getI1Type(), available, lastOffset,
            ComparePredicate::Ge));
      }
      start = builder.create<block::BinaryOp>(
          location, builder.getIndexType(), start, modulus,
          BinaryOperator::Remainder);
      Value remaining = builder.create<block::BinaryOp>(
          location, builder.getIndexType(), modulus, start,
          BinaryOperator::Subtract);
      // A tile that crosses the modulo boundary keeps its original gather.
      Value contained = builder.create<block::CompareOp>(
          location, builder.getI1Type(), remaining, extent, ComparePredicate::Ge);
      if (uniformPeriodicAlignment)
        contained = builder.create<block::BinaryOp>(
            location, builder.getI1Type(), uniformPeriodicAlignment, contained,
            BinaryOperator::LogicalOr);
      require(contained);
    }
    FailureOr<Value> index = tileIndex(builder, owner->getLoc(), start,
                                       extent);
    if (succeeded(index)) {
      result.values.push_back(*index);
      continue;
    }
    if (!allowDynamicAlignment || !block::IndexRelations().positive(extent))
      return failure();
    result.values.push_back(builder.create<block::BinaryOp>(
        owner->getLoc(), builder.getIndexType(), start, extent,
        BinaryOperator::FloorDivide));
    Value remainder = builder.create<block::BinaryOp>(
        owner->getLoc(), builder.getIndexType(), start, extent,
        BinaryOperator::Remainder);
    Value zero = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 0);
    Value aligned = builder.create<block::CompareOp>(
        owner->getLoc(), builder.getI1Type(), remainder, zero,
        ComparePredicate::Eq);
    if (block::IndexRelations().powerOfTwo(extent)) {
      auto factors = uniformAlignmentFactors(start, extent);
      if (succeeded(factors)) {
        if (factors->empty())
          continue;
        Value uniform;
        for (Value factor : *factors) {
          Value modulus = builder.create<block::BinaryOp>(
              owner->getLoc(), builder.getIndexType(), factor,
              extent, BinaryOperator::Remainder);
          Value condition = builder.create<block::CompareOp>(
              owner->getLoc(), builder.getI1Type(), modulus, zero,
              ComparePredicate::Eq);
          uniform = uniform ? Value(builder.create<block::BinaryOp>(
                                  owner->getLoc(), builder.getI1Type(), uniform,
                                  condition, BinaryOperator::LogicalAnd))
                            : condition;
        }
        // Specialization can discard the gather branch for the entire loop.
        // Otherwise retain the original per-origin alignment predicate.
        aligned = builder.create<block::BinaryOp>(
            owner->getLoc(), builder.getI1Type(), uniform, aligned,
            BinaryOperator::LogicalOr);
      }
    }
    if (uniformPeriodicAlignment)
      aligned = builder.create<block::BinaryOp>(
          owner->getLoc(), builder.getI1Type(), uniformPeriodicAlignment, aligned,
          BinaryOperator::LogicalOr);
    require(aligned);
  }
  return result;
}

Value materializeFullTileCondition(OpBuilder &builder, Location location,
                                   Value resource, block::FragmentType tile) {
  Value fullTiles = builder.create<arith::ConstantIntOp>(location, 1, 1);
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  for (auto [axis, attribute] : llvm::enumerate(tile.getShape())) {
    auto extent = cast<block::PhysicalExprAttr>(attribute);
    if (extent.getKind() == block::PhysicalExprKind::Constant && extent.getValue() == 1)
      continue;
    Value size = builder.create<block::DimOp>(location, builder.getIndexType(), resource, axis);
    Value width = builder.create<block::PhysicalExprOp>(location, builder.getIndexType(), extent);
    Value remainder = builder.create<block::BinaryOp>(
        location, builder.getIndexType(), size, width, BinaryOperator::Remainder);
    Value divisible = builder.create<block::CompareOp>(
        location, builder.getI1Type(), remainder, zero, ComparePredicate::Eq);
    fullTiles = builder.create<block::BinaryOp>(
        location, builder.getI1Type(), fullTiles, divisible, BinaryOperator::LogicalAnd);
  }
  return fullTiles;
}

} // namespace intent::cutile
