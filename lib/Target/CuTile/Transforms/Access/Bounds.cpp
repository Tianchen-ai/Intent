#include "Bounds.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;

namespace intent::cutile {

std::optional<int64_t> dimensionIdentity(BlockArgument argument) {
  auto kernel = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!kernel || argument.getOwner() != &kernel.getBody().front())
    return std::nullopt;
  auto binding = gpu::getArgumentBinding(argument);
  if (!binding || binding.getKind() != gpu::ArgumentKind::Dimension)
    return std::nullopt;
  return binding.getDimension().getInt();
}

FailureOr<SmallVector<Value>> uniformAlignmentFactors(
    Value value, Value divisor, unsigned depth) {
  if (!value || depth >= 32)
    return failure();
  value = stripIndexIdentities(value);
  divisor = stripIndexIdentities(divisor);
  if (gpu::IndexRelations().multipleOf(value, divisor))
    return SmallVector<Value>{};
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    if (!integer)
      return failure();
    return integer.getValue().isZero() ? SmallVector<Value>{}
                                       : SmallVector<Value>{value};
  }
  if (value.getDefiningOp<gpu::ParameterOp>() ||
      value.getDefiningOp<gpu::DimOp>() ||
      value.getDefiningOp<gpu::PhysicalExprOp>())
    return SmallVector<Value>{value};
  auto combine = [&](Value lhs, Value rhs) -> FailureOr<SmallVector<Value>> {
    auto left = uniformAlignmentFactors(lhs, divisor, depth + 1);
    auto right = uniformAlignmentFactors(rhs, divisor, depth + 1);
    if (failed(left) || failed(right))
      return failure();
    for (Value factor : *right)
      if (!llvm::is_contained(*left, factor))
        left->push_back(factor);
    return std::move(*left);
  };
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (dimensionIdentity(argument))
      return SmallVector<Value>{value};
    auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (loop && argument == loop.getInductionVar() &&
        gpu::IndexRelations().positive(loop.getStep()))
      return combine(loop.getLowerBound(), loop.getStep());
    return failure();
  }
  auto binary = value.getDefiningOp<gpu::BinaryOp>();
  if (!binary)
    return failure();
  switch (binary.getOperatorKind()) {
  case BinaryOperator::Add:
  case BinaryOperator::Subtract:
  case BinaryOperator::Minimum:
  case BinaryOperator::Maximum:
    return combine(binary.getLhs(), binary.getRhs());
  case BinaryOperator::Multiply: {
    auto lhs = uniformAlignmentFactors(binary.getLhs(), divisor, depth + 1);
    auto rhs = uniformAlignmentFactors(binary.getRhs(), divisor, depth + 1);
    if (succeeded(lhs) && (failed(rhs) || lhs->size() <= rhs->size()))
      return std::move(*lhs);
    return rhs;
  }
  default:
    return failure();
  }
}

bool isAlignedPeriodicTile(Value start, Value extent, Value period) {
  gpu::IndexRelations relations;
  // Dividing the period into complete aligned windows keeps one quotient.
  return relations.alignedUnitWindow(stripIndexIdentities(start),
                                     stripIndexIdentities(extent)) &&
         relations.positive(period) && relations.multipleOf(period, extent);
}

scf::ForOp completeAlignedTileLoop(Value start, Value extent) {
  auto induction = dyn_cast<BlockArgument>(stripIndexIdentities(start));
  auto loop = induction
                  ? dyn_cast_or_null<scf::ForOp>(
                        induction.getOwner()->getParentOp())
                  : scf::ForOp();
  if (!loop || induction != loop.getInductionVar() ||
      !gpu::IndexRelations().powerOfTwo(extent) ||
      !gpu::samePhysicalScalarExpression(loop.getStep(), extent) ||
      !gpu::queryNonNegativeIndexUpperBound(loop.getLowerBound()) ||
      !gpu::IndexRelations().multipleOf(loop.getLowerBound(), extent) ||
      !gpu::IndexRelations().multipleOf(loop.getUpperBound(), extent))
    return {};
  // Both endpoints are tile aligned. Every executing iteration is nonnegative
  // and complete; the final increment reaches the representable upper bound.
  return loop;
}

bool valueIsDimension(Value value, int64_t dimension, func::FuncOp kernel) {
  value = stripIndexIdentities(value);
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() != &kernel.getBody().front())
      return false;
    std::optional<int64_t> identity = dimensionIdentity(argument);
    return identity && *identity == dimension;
  }
  if (auto dim = value.getDefiningOp<gpu::DimOp>()) {
    ArrayRef<int64_t> dimensions =
        dim.getView().getType().getLayout().getDimensionIds().asArrayRef();
    return dim.getAxis() < dimensions.size() &&
           dimensions[dim.getAxis()] == dimension;
  }
  return false;
}

Value dimensionValue(func::FuncOp kernel, int64_t dimension) {
  Value result;
  for (BlockArgument argument : kernel.getArguments()) {
    std::optional<int64_t> identity = dimensionIdentity(argument);
    if (!identity || *identity != dimension)
      continue;
    if (result && result != argument)
      return {};
    result = argument;
  }
  return result;
}

bool isLogicalSubregionDistance(Value value, gpu::MakeRangeOp range) {
  value = stripIndexIdentities(value);
  auto distance = value.getDefiningOp<gpu::BinaryOp>();
  if (distance && distance.getOperatorKind() == BinaryOperator::Subtract &&
      gpu::samePhysicalScalarExpression(distance.getLhs(),
                                        range.getLogicalStop()) &&
      gpu::samePhysicalScalarExpression(distance.getRhs(),
                                        range.getLogicalStart()))
    return true;
  auto multiply = value.getDefiningOp<gpu::BinaryOp>();
  if (!multiply || multiply.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  auto flooredDistance = [&](Value quotient, Value divisor) {
    auto division = quotient.getDefiningOp<gpu::BinaryOp>();
    return division &&
           division.getOperatorKind() == BinaryOperator::FloorDivide &&
           gpu::samePhysicalScalarExpression(division.getRhs(), divisor) &&
           gpu::IndexRelations().positive(divisor) &&
           isLogicalSubregionDistance(division.getLhs(), range);
  };
  return flooredDistance(multiply.getLhs(), multiply.getRhs()) ||
         flooredDistance(multiply.getRhs(), multiply.getLhs());
}

bool rangeStartsInsideLogicalSubregion(gpu::MakeRangeOp range) {
  Value start = stripIndexIdentities(range.getStart());
  if (gpu::samePhysicalScalarExpression(start, range.getLogicalStart()))
    return true;
  auto add = start.getDefiningOp<gpu::BinaryOp>();
  if (!add || add.getOperatorKind() != BinaryOperator::Add)
    return false;
  auto check = [&](Value base, Value offset) {
    if (!gpu::samePhysicalScalarExpression(base, range.getLogicalStart()))
      return false;
    auto argument = dyn_cast<BlockArgument>(stripIndexIdentities(offset));
    auto loop =
        argument
            ? dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())
            : scf::ForOp();
    return loop && argument == loop.getInductionVar() &&
           gpu::IndexRelations().nonnegative(loop.getLowerBound()) &&
           gpu::IndexRelations().positive(loop.getStep()) &&
           isLogicalSubregionDistance(loop.getUpperBound(), range);
  };
  return check(add.getLhs(), add.getRhs()) ||
         check(add.getRhs(), add.getLhs());
}

bool isBlockedWorksetOrigin(Value value, Value block, int64_t dimension) {
  auto multiply = value.getDefiningOp<gpu::BinaryOp>();
  if (!multiply || multiply.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  Value coordinate;
  if (multiply.getLhs() == block)
    coordinate = multiply.getRhs();
  else if (multiply.getRhs() == block)
    coordinate = multiply.getLhs();
  else
    return false;
  auto mapping = gpu::queryDecodedCoordinate(coordinate);
  if (!mapping || mapping->axis >= mapping->extents.size())
    return false;
  auto expression = gpu::queryLaunchExpression(mapping->extents[mapping->axis]);
  auto parameter = block.getDefiningOp<gpu::ParameterOp>();
  if (!expression || !parameter || expression.getOperands().size() != 2)
    return false;
  auto isBlock = [&](gpu::PhysicalExprAttr operand) {
    return operand.getKind() == gpu::PhysicalExprKind::Parameter &&
           operand.getParameterReference() == parameter.getReference();
  };
  auto isDimension = [&](gpu::PhysicalExprAttr operand) {
    return operand.getKind() == gpu::PhysicalExprKind::Dimension &&
           operand.getValue() == dimension;
  };
  auto numerator = cast<gpu::PhysicalExprAttr>(expression.getOperands()[0]);
  auto denominator = cast<gpu::PhysicalExprAttr>(expression.getOperands()[1]);
  if (!isBlock(denominator)) return false;
  if (expression.getKind() == gpu::PhysicalExprKind::CeilDiv)
    return isDimension(numerator);
  // Ownership formation spells the runtime tile count as (D + (P - 1)) // P.
  // Read that current SSA relation after execution-group lowering, rather than
  // attaching the old launch schema to a pure coordinate calculation.
  if (expression.getKind() != gpu::PhysicalExprKind::FloorDiv ||
      numerator.getKind() != gpu::PhysicalExprKind::Add ||
      numerator.getOperands().size() != 2)
    return false;
  auto adjusted = [&](gpu::PhysicalExprAttr logical,
                      gpu::PhysicalExprAttr padding) {
    return isDimension(logical) &&
           padding.getKind() == gpu::PhysicalExprKind::Subtract &&
           padding.getOperands().size() == 2 &&
           isBlock(cast<gpu::PhysicalExprAttr>(padding.getOperands()[0])) &&
           gpu::constantPhysicalExpression(
               cast<gpu::PhysicalExprAttr>(padding.getOperands()[1])) == 1;
  };
  auto lhs = cast<gpu::PhysicalExprAttr>(numerator.getOperands()[0]);
  auto rhs = cast<gpu::PhysicalExprAttr>(numerator.getOperands()[1]);
  return adjusted(lhs, rhs) || adjusted(rhs, lhs);
}

std::optional<int64_t> constantTileOrigin(gpu::MakeRangeOp range,
                                          ArrayRef<Value> offsets) {
  std::optional<int64_t> result = gpu::IndexRelations().constant(range.getStart());
  if (!result)
    return std::nullopt;
  for (Value offset : offsets) {
    std::optional<int64_t> constant = gpu::IndexRelations().constant(offset);
    if (!constant)
      return std::nullopt;
    int64_t sum;
    if (llvm::AddOverflow(*result, *constant, sum))
      return std::nullopt;
    *result = sum;
  }
  return result;
}

bool constantOriginInView(int64_t origin, gpu::ViewType view,
                          unsigned resourceAxis, func::FuncOp kernel) {
  if (origin < 0 || resourceAxis >= view.getLayout().getExtents().size())
    return false;
  auto extent = dyn_cast<gpu::PhysicalExprAttr>(
      view.getLayout().getExtents()[resourceAxis]);
  std::optional<int64_t> constant =
      gpu::IndexRelations().constant(extent, kernel);
  return constant && origin < *constant;
}

bool rangeOriginInView(gpu::MakeRangeOp range, ArrayRef<Value> offsets,
                       gpu::ViewType view, unsigned resourceAxis,
                       int64_t dimension, func::FuncOp kernel) {
  bool zeroOffset = llvm::all_of(
      offsets, [](Value offset) { return isProvably(offset, 0); });
  Value logicalDimension = dimensionValue(kernel, dimension);
  if (zeroOffset && range->hasAttr(gpu::sourceSubregionAttr) &&
      logicalDimension && gpu::IndexRelations().nonnegative(range.getLogicalStart()) &&
      gpu::IndexRelations().atMost(range.getLogicalStop(), logicalDimension) &&
      rangeStartsInsideLogicalSubregion(range))
    return true;
  if (zeroOffset && range->hasAttr(gpu::programBoundedOriginAttr) &&
      !range->hasAttr(gpu::sourceSubregionAttr) &&
      isProvably(range.getLogicalStart(), 0) &&
      valueIsDimension(range.getLogicalStop(), dimension, kernel))
    return true;
  if (zeroOffset &&
      valueIsDimension(range.getLogicalStop(), dimension, kernel)) {
    Value start = stripIndexIdentities(range.getStart());
    if (isProvably(range.getLogicalStart(), 0) &&
        isBlockedWorksetOrigin(start, range.getExtent(), dimension))
      return true;
    if (auto coordinate = start.getDefiningOp<gpu::WorksetCoordinateOp>())
      if (coordinate.getDimensionId() == static_cast<uint64_t>(dimension))
        return true;
    if (auto argument = dyn_cast<BlockArgument>(start)) {
      auto loop = dyn_cast_or_null<scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (loop && argument == loop.getInductionVar() &&
          isProvably(range.getLogicalStart(), 0) &&
          gpu::IndexRelations().nonnegative(loop.getLowerBound()) &&
          gpu::IndexRelations().atMost(loop.getUpperBound(), range.getLogicalStop()))
        return true;
    }
    auto parameter = range.getExtent().getDefiningOp<gpu::ParameterOp>();
    if (isProvably(start, 0) && isProvably(range.getLogicalStart(), 0) &&
        parameter &&
        parameter.getDeclaration().getRole() ==
            gpu::ParameterRole::FullCoverage) {
      gpu::PhysicalParameterBinding binding =
          gpu::queryParameterBinding(parameter.getDeclaration());
      if (binding.isExact() && binding.dimension &&
          *binding.dimension == dimension)
        return true;
    }
  }
  std::optional<int64_t> origin = constantTileOrigin(range, offsets);
  return origin &&
         constantOriginInView(*origin, view, resourceAxis, kernel);
}

bool scalarOriginInView(Value index, gpu::ViewType view,
                        unsigned resourceAxis, int64_t dimension,
                        func::FuncOp kernel) {
  index = stripIndexIdentities(index);
  if (auto coordinate = index.getDefiningOp<gpu::WorksetCoordinateOp>())
    if (coordinate.getDimensionId() == static_cast<uint64_t>(dimension))
      return true;
  if (auto argument = dyn_cast<BlockArgument>(index)) {
    auto loop =
        dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    Value logicalDimension = dimensionValue(kernel, dimension);
    if (loop && argument == loop.getInductionVar() && logicalDimension &&
        gpu::IndexRelations().nonnegative(loop.getLowerBound()) &&
        gpu::IndexRelations().positive(loop.getStep()) &&
        gpu::IndexRelations().atMost(loop.getUpperBound(), logicalDimension))
      return true;
  }
  std::optional<int64_t> origin = gpu::IndexRelations().constant(index);
  return origin &&
         constantOriginInView(*origin, view, resourceAxis, kernel);
}

bool scalarCoordinatesInView(ValueRange coordinates, gpu::ViewType view,
                             func::FuncOp kernel) {
  ArrayRef<int64_t> dimensions =
      view.getLayout().getDimensionIds().asArrayRef();
  if (coordinates.size() != view.getRank() ||
      dimensions.size() != view.getRank())
    return false;
  for (auto [axis, coordinate] : llvm::enumerate(coordinates))
    if (!scalarOriginInView(coordinate, view, axis, dimensions[axis], kernel))
      return false;
  return true;
}

} // namespace intent::cutile
