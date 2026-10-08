#include "TileIndices.h"
#include "Bounds.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
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
  std::optional<int64_t> startConstant = gpu::IndexRelations().constant(start);
  std::optional<int64_t> extentConstant = gpu::IndexRelations().constant(extent);
  if (startConstant && extentConstant && *startConstant >= 0 &&
      *extentConstant > 0 && *startConstant % *extentConstant == 0)
    return builder
        .create<arith::ConstantIndexOp>(location,
                                        *startConstant / *extentConstant)
        .getResult();
  if (isProvably(start, 0))
    return builder.create<arith::ConstantIndexOp>(location, 0).getResult();
  if (auto coordinate = start.getDefiningOp<gpu::WorksetCoordinateOp>())
    return tileIndex(builder, location, coordinate.getCoordinate(), extent);
  if (auto binary = start.getDefiningOp<gpu::BinaryOp>()) {
    if (binary.getOperatorKind() == BinaryOperator::Multiply) {
      if (gpu::samePhysicalScalarExpression(binary.getLhs(), extent))
        return binary.getRhs();
      if (gpu::samePhysicalScalarExpression(binary.getRhs(), extent))
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
          gpu::samePhysicalScalarExpression(loop.getLowerBound(),
                                            binary.getRhs()) &&
          gpu::samePhysicalScalarExpression(loop.getStep(), extent))
        return Value(builder.create<gpu::BinaryOp>(
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
      gpu::samePhysicalScalarExpression(loop.getStep(), extent) &&
      gpu::IndexRelations().multipleOf(loop.getLowerBound(), extent))
    return Value(builder.create<gpu::BinaryOp>(
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
    return builder.create<gpu::BinaryOp>(location, builder.getIndexType(),
        axis.scalarIndex, divisor, BinaryOperator::FloorDivide);
  }
  Value origin;
  auto add = [&](Value value, int64_t scale) {
    if (scale != 1) {
      Value coefficient = builder.create<arith::ConstantIndexOp>(location, scale);
      value = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), value, coefficient,
          BinaryOperator::Multiply);
    }
    origin = origin ? Value(builder.create<gpu::BinaryOp>(
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

} // namespace

FailureOr<Value> materializeTileOriginGuard(
    OpBuilder &builder, Operation *owner, Value resource,
    NativeTileAccessPlan &plan,
    const gpu::PhysicalAccessBoundaryFact &boundary) {
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
    if (axisPlan.modulus)
      origin = builder.create<gpu::BinaryOp>(
          owner->getLoc(), builder.getIndexType(), origin,
          materializeTileModulus(builder, owner->getLoc(), axisPlan),
          BinaryOperator::Remainder);
    if (!zero)
      zero = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 0);
    Value extent = builder.create<gpu::DimOp>(
        owner->getLoc(), builder.getIndexType(), resource, axis);
    Value nonNegative = builder.create<gpu::CompareOp>(
        owner->getLoc(), builder.getI1Type(), origin, zero,
        ComparePredicate::Ge);
    Value belowExtent = builder.create<gpu::CompareOp>(
        owner->getLoc(), builder.getI1Type(), origin, extent,
        ComparePredicate::Lt);
    Value axisCondition = builder.create<gpu::BinaryOp>(
        owner->getLoc(), builder.getI1Type(), nonNegative, belowExtent,
        BinaryOperator::LogicalAnd);
    if (gpu::PhysicalExprAttr upper =
            gpu::queryNonNegativeIndexUpperBound(origin)) {
      Value bound = builder.create<gpu::PhysicalExprOp>(
          owner->getLoc(), builder.getIndexType(), upper);
      Value allOriginsInBounds = builder.create<gpu::CompareOp>(
          owner->getLoc(), builder.getI1Type(), bound, extent,
          ComparePredicate::Lt);
      // Specialization can discharge the whole mapped domain. Otherwise the
      // original per-origin predicate still determines exactly the same access.
      axisCondition = builder.create<gpu::BinaryOp>(
          owner->getLoc(), builder.getI1Type(), allOriginsInBounds, axisCondition,
          BinaryOperator::LogicalOr);
    }
    condition = condition
                    ? Value(builder.create<gpu::BinaryOp>(
                          owner->getLoc(), builder.getI1Type(), condition,
                          axisCondition, BinaryOperator::LogicalAnd))
                    : axisCondition;
  }
  return condition;
}

Value materializeFullRangeGuard(
    OpBuilder &builder, Location location,
    const gpu::PhysicalAccessBoundaryFact &boundary) {
  Value guard;
  for (auto [range, upper] : boundary.rangeBounds) {
    if (auto loop = completeAlignedTileLoop(range.getStart(), range.getExtent()))
      if (gpu::samePhysicalScalarExpression(loop.getUpperBound(), upper))
        continue;
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value origin = range.getStart();
    Value nonnegativeOrigin = builder.create<gpu::CompareOp>(
        location, builder.getI1Type(), origin, zero, ComparePredicate::Ge);
    Value nonnegativeUpper = builder.create<gpu::CompareOp>(
        location, builder.getI1Type(), upper, zero, ComparePredicate::Ge);
    Value nonnegative = builder.create<gpu::BinaryOp>(
        location, builder.getI1Type(), nonnegativeOrigin, nonnegativeUpper,
        BinaryOperator::LogicalAnd);
    // With nonnegative endpoints, subtraction cannot overflow. A failed guard
    // keeps the original predicated gather, including a partial logical tile.
    Value remaining = builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), upper, origin,
        BinaryOperator::Subtract);
    Value fullRange = builder.create<gpu::CompareOp>(
        location, builder.getI1Type(), remaining, range.getExtent(),
        ComparePredicate::Ge);
    Value condition = builder.create<gpu::BinaryOp>(
        location, builder.getI1Type(), nonnegative, fullRange,
        BinaryOperator::LogicalAnd);
    guard = guard ? Value(builder.create<gpu::BinaryOp>(
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
                           ? Value(builder.create<gpu::BinaryOp>(
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
        if (!gpu::IndexRelations().positive(extent))
          return failure();
        Value divisor = builder.create<arith::ConstantIndexOp>(owner->getLoc(), axis.divisor);
        if (isAlignedPeriodicTile(axis.scalarIndex, extent, divisor)) {
          result.values.push_back(index);
          continue;
        }
        Value one = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 1);
        Value offset = builder.create<gpu::BinaryOp>(owner->getLoc(),
            builder.getIndexType(), extent, one, BinaryOperator::Subtract);
        Value last = builder.create<gpu::BinaryOp>(owner->getLoc(),
            builder.getIndexType(), axis.scalarIndex, offset, BinaryOperator::Add);
        Value lastIndex = builder.create<gpu::BinaryOp>(owner->getLoc(),
            builder.getIndexType(), last, divisor, BinaryOperator::FloorDivide);
        // Positive extent and divisor make equal endpoint quotients sufficient;
        // signed wraparound would change the quotient's sign and fail the guard.
        require(builder.create<gpu::CompareOp>(owner->getLoc(), builder.getI1Type(),
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
      extent = builder.create<gpu::PhysicalExprOp>(
          owner->getLoc(), builder.getIndexType(),
          cast<gpu::PhysicalExprAttr>(plan.resourceType.getShape()[resourceAxis]));
      Value inner = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 1);
      Value one = inner;
      // A varying axis is contiguous only when its stride equals the product
      // of the inner tile extents. Unit axes contribute an origin, not a gap.
      for (auto [range, stride] : llvm::reverse(axis.ranges)) {
        Value coefficient = builder.create<arith::ConstantIndexOp>(
            owner->getLoc(), stride);
        Value unit = builder.create<gpu::CompareOp>(
            owner->getLoc(), builder.getI1Type(), range.getExtent(), one,
            ComparePredicate::Eq);
        Value contiguous = builder.create<gpu::CompareOp>(
            owner->getLoc(), builder.getI1Type(), coefficient, inner,
            ComparePredicate::Eq);
        require(builder.create<gpu::BinaryOp>(
            owner->getLoc(), builder.getI1Type(), unit, contiguous,
            BinaryOperator::LogicalOr));
        inner = builder.create<gpu::BinaryOp>(
            owner->getLoc(), builder.getIndexType(), inner, range.getExtent(),
            BinaryOperator::Multiply);
      }
    }
    if (axis.modulus) {
      if (!gpu::IndexRelations().positive(extent))
        return failure();
      Location location = owner->getLoc();
      Value modulus = materializeTileModulus(builder, location, axis);
      if (isAlignedPeriodicTile(start, extent, modulus) &&
          completeAlignedTileLoop(start, extent)) {
        Value remainder = builder.create<gpu::BinaryOp>(
            location, builder.getIndexType(), start, modulus,
            BinaryOperator::Remainder);
        result.values.push_back(builder.create<gpu::BinaryOp>(
            location, builder.getIndexType(), remainder, extent,
            BinaryOperator::FloorDivide));
        continue;
      }
      Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value maximum = builder.create<arith::ConstantIndexOp>(
          location, std::numeric_limits<int64_t>::max());
      require(builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), modulus, zero, ComparePredicate::Gt));
      require(builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), start, zero, ComparePredicate::Ge));
      Value available = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), maximum, start,
          BinaryOperator::Subtract);
      Value lastOffset = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), extent, one,
          BinaryOperator::Subtract);
      require(builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), available, lastOffset,
          ComparePredicate::Ge));
      start = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), start, modulus,
          BinaryOperator::Remainder);
      Value remaining = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), modulus, start,
          BinaryOperator::Subtract);
      // A tile that crosses the modulo boundary keeps its original gather.
      require(builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), remaining, extent, ComparePredicate::Ge));
    }
    FailureOr<Value> index = tileIndex(builder, owner->getLoc(), start,
                                       extent);
    if (succeeded(index)) {
      result.values.push_back(*index);
      continue;
    }
    if (!allowDynamicAlignment || !gpu::IndexRelations().positive(extent))
      return failure();
    result.values.push_back(builder.create<gpu::BinaryOp>(
        owner->getLoc(), builder.getIndexType(), start, extent,
        BinaryOperator::FloorDivide));
    Value remainder = builder.create<gpu::BinaryOp>(
        owner->getLoc(), builder.getIndexType(), start, extent,
        BinaryOperator::Remainder);
    Value zero = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 0);
    Value aligned = builder.create<gpu::CompareOp>(
        owner->getLoc(), builder.getI1Type(), remainder, zero,
        ComparePredicate::Eq);
    if (gpu::IndexRelations().powerOfTwo(extent)) {
      auto factors = uniformAlignmentFactors(start, extent);
      if (succeeded(factors)) {
        if (factors->empty())
          continue;
        Value uniform;
        for (Value factor : *factors) {
          Value modulus = builder.create<gpu::BinaryOp>(
              owner->getLoc(), builder.getIndexType(), factor,
              extent, BinaryOperator::Remainder);
          Value condition = builder.create<gpu::CompareOp>(
              owner->getLoc(), builder.getI1Type(), modulus, zero,
              ComparePredicate::Eq);
          uniform = uniform ? Value(builder.create<gpu::BinaryOp>(
                                  owner->getLoc(), builder.getI1Type(), uniform,
                                  condition, BinaryOperator::LogicalAnd))
                            : condition;
        }
        // Specialization can discard the gather branch for the entire loop.
        // Otherwise retain the original per-origin alignment predicate.
        aligned = builder.create<gpu::BinaryOp>(
            owner->getLoc(), builder.getI1Type(), uniform, aligned,
            BinaryOperator::LogicalOr);
      }
    }
    require(aligned);
  }
  return result;
}

Value materializeFullTileCondition(OpBuilder &builder, Location location,
                                   Value resource, gpu::FragmentType tile) {
  Value fullTiles = builder.create<arith::ConstantIntOp>(location, 1, 1);
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  for (auto [axis, attribute] : llvm::enumerate(tile.getShape())) {
    auto extent = cast<gpu::PhysicalExprAttr>(attribute);
    if (extent.getKind() == gpu::PhysicalExprKind::Constant && extent.getValue() == 1)
      continue;
    Value size = builder.create<gpu::DimOp>(location, builder.getIndexType(), resource, axis);
    Value width = builder.create<gpu::PhysicalExprOp>(location, builder.getIndexType(), extent);
    Value remainder = builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), size, width, BinaryOperator::Remainder);
    Value divisible = builder.create<gpu::CompareOp>(
        location, builder.getI1Type(), remainder, zero, ComparePredicate::Eq);
    fullTiles = builder.create<gpu::BinaryOp>(
        location, builder.getI1Type(), fullTiles, divisible, BinaryOperator::LogicalAnd);
  }
  return fullTiles;
}

} // namespace intent::cutile
