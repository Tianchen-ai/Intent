#include "Bounds.h"
#include "Intent/Dialect/Block/Analysis/IndexRelations.h"
#include "Intent/Dialect/Block/Analysis/ProgramInterface.h"
#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "Intent/Dialect/Block/IR/PhysicalExpressions.h"
#include "Intent/Dialect/Block/IR/Program.h"
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
  auto binding = block::getArgumentBinding(argument);
  if (!binding || binding.getKind() != block::ArgumentKind::Dimension)
    return std::nullopt;
  return binding.getDimension().getInt();
}

FailureOr<SmallVector<Value>> uniformAlignmentFactors(
    Value value, Value divisor, unsigned depth) {
  if (!value || depth >= 32)
    return failure();
  value = stripIndexIdentities(value);
  divisor = stripIndexIdentities(divisor);
  if (block::IndexRelations().multipleOf(value, divisor))
    return SmallVector<Value>{};
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    if (!integer)
      return failure();
    return integer.getValue().isZero() ? SmallVector<Value>{}
                                       : SmallVector<Value>{value};
  }
  if (value.getDefiningOp<block::ParameterOp>() ||
      value.getDefiningOp<block::DimOp>() ||
      value.getDefiningOp<block::PhysicalExprOp>())
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
        block::IndexRelations().positive(loop.getStep()))
      return combine(loop.getLowerBound(), loop.getStep());
    return failure();
  }
  auto binary = value.getDefiningOp<block::BinaryOp>();
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
  block::IndexRelations relations;
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
      !block::IndexRelations().powerOfTwo(extent) ||
      !block::samePhysicalScalarExpression(loop.getStep(), extent) ||
      !block::queryNonNegativeIndexUpperBound(loop.getLowerBound()) ||
      !block::IndexRelations().multipleOf(loop.getLowerBound(), extent) ||
      !block::IndexRelations().multipleOf(loop.getUpperBound(), extent))
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
  if (auto dim = value.getDefiningOp<block::DimOp>()) {
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

bool isLogicalSubregionDistance(Value value, block::MakeRangeOp range) {
  value = stripIndexIdentities(value);
  auto distance = value.getDefiningOp<block::BinaryOp>();
  if (distance && distance.getOperatorKind() == BinaryOperator::Subtract &&
      block::samePhysicalScalarExpression(distance.getLhs(),
                                        range.getLogicalStop()) &&
      block::samePhysicalScalarExpression(distance.getRhs(),
                                        range.getLogicalStart()))
    return true;
  auto multiply = value.getDefiningOp<block::BinaryOp>();
  if (!multiply || multiply.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  auto flooredDistance = [&](Value quotient, Value divisor) {
    auto division = quotient.getDefiningOp<block::BinaryOp>();
    return division &&
           division.getOperatorKind() == BinaryOperator::FloorDivide &&
           block::samePhysicalScalarExpression(division.getRhs(), divisor) &&
           block::IndexRelations().positive(divisor) &&
           isLogicalSubregionDistance(division.getLhs(), range);
  };
  return flooredDistance(multiply.getLhs(), multiply.getRhs()) ||
         flooredDistance(multiply.getRhs(), multiply.getLhs());
}

bool rangeStartsInsideLogicalSubregion(block::MakeRangeOp range) {
  Value start = stripIndexIdentities(range.getStart());
  if (block::samePhysicalScalarExpression(start, range.getLogicalStart()))
    return true;
  auto add = start.getDefiningOp<block::BinaryOp>();
  if (!add || add.getOperatorKind() != BinaryOperator::Add)
    return false;
  auto check = [&](Value base, Value offset) {
    if (!block::samePhysicalScalarExpression(base, range.getLogicalStart()))
      return false;
    auto argument = dyn_cast<BlockArgument>(stripIndexIdentities(offset));
    auto loop =
        argument
            ? dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())
            : scf::ForOp();
    return loop && argument == loop.getInductionVar() &&
           block::IndexRelations().nonnegative(loop.getLowerBound()) &&
           block::IndexRelations().positive(loop.getStep()) &&
           isLogicalSubregionDistance(loop.getUpperBound(), range);
  };
  return check(add.getLhs(), add.getRhs()) ||
         check(add.getRhs(), add.getLhs());
}

std::optional<int64_t> constantTileOrigin(block::MakeRangeOp range,
                                          ArrayRef<Value> offsets) {
  std::optional<int64_t> result = block::IndexRelations().constant(range.getStart());
  if (!result)
    return std::nullopt;
  for (Value offset : offsets) {
    std::optional<int64_t> constant = block::IndexRelations().constant(offset);
    if (!constant)
      return std::nullopt;
    int64_t sum;
    if (llvm::AddOverflow(*result, *constant, sum))
      return std::nullopt;
    *result = sum;
  }
  return result;
}

bool constantOriginInView(int64_t origin, block::ViewType view,
                          unsigned resourceAxis, func::FuncOp kernel) {
  if (origin < 0 || resourceAxis >= view.getLayout().getExtents().size())
    return false;
  auto extent = dyn_cast<block::PhysicalExprAttr>(
      view.getLayout().getExtents()[resourceAxis]);
  std::optional<int64_t> constant =
      block::IndexRelations().constant(extent, kernel);
  return constant && origin < *constant;
}

bool rangeOriginInView(block::MakeRangeOp range, ArrayRef<Value> offsets,
                       block::ViewType view, unsigned resourceAxis,
                       int64_t dimension, func::FuncOp kernel) {
  bool zeroOffset = llvm::all_of(
      offsets, [](Value offset) { return isProvably(offset, 0); });
  Value logicalDimension = dimensionValue(kernel, dimension);
  if (zeroOffset && range->hasAttr(block::sourceSubregionAttr) &&
      logicalDimension && block::IndexRelations().nonnegative(range.getLogicalStart()) &&
      block::IndexRelations().atMost(range.getLogicalStop(), logicalDimension) &&
      rangeStartsInsideLogicalSubregion(range))
    return true;
  if (zeroOffset && range->hasAttr(block::programBoundedOriginAttr) &&
      !range->hasAttr(block::sourceSubregionAttr) &&
      isProvably(range.getLogicalStart(), 0) &&
      valueIsDimension(range.getLogicalStop(), dimension, kernel))
    return true;
  if (zeroOffset &&
      valueIsDimension(range.getLogicalStop(), dimension, kernel)) {
    Value start = stripIndexIdentities(range.getStart());
    if (isProvably(range.getLogicalStart(), 0) &&
        block::IndexRelations().mappedTileOrigin(start, range.getExtent(), dimension))
      return true;
    if (auto coordinate = start.getDefiningOp<block::WorksetCoordinateOp>())
      if (coordinate.getDimensionId() == static_cast<uint64_t>(dimension))
        return true;
    if (auto argument = dyn_cast<BlockArgument>(start)) {
      auto loop = dyn_cast_or_null<scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (loop && argument == loop.getInductionVar() &&
          isProvably(range.getLogicalStart(), 0) &&
          block::IndexRelations().nonnegative(loop.getLowerBound()) &&
          block::IndexRelations().atMost(loop.getUpperBound(), range.getLogicalStop()))
        return true;
    }
    auto parameter = range.getExtent().getDefiningOp<block::ParameterOp>();
    if (isProvably(start, 0) && isProvably(range.getLogicalStart(), 0) &&
        parameter &&
        parameter.getDeclaration().getRole() ==
            block::ParameterRole::FullCoverage) {
      block::PhysicalParameterBinding binding =
          block::queryParameterBinding(parameter.getDeclaration());
      if (binding.isExact() && binding.dimension &&
          *binding.dimension == dimension)
        return true;
    }
  }
  std::optional<int64_t> origin = constantTileOrigin(range, offsets);
  return origin &&
         constantOriginInView(*origin, view, resourceAxis, kernel);
}

bool scalarOriginInView(Value index, block::ViewType view,
                        unsigned resourceAxis, int64_t dimension,
                        func::FuncOp kernel) {
  index = stripIndexIdentities(index);
  if (auto coordinate = index.getDefiningOp<block::WorksetCoordinateOp>())
    if (coordinate.getDimensionId() == static_cast<uint64_t>(dimension))
      return true;
  if (auto argument = dyn_cast<BlockArgument>(index)) {
    auto loop =
        dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    Value logicalDimension = dimensionValue(kernel, dimension);
    if (loop && argument == loop.getInductionVar() && logicalDimension &&
        block::IndexRelations().nonnegative(loop.getLowerBound()) &&
        block::IndexRelations().positive(loop.getStep()) &&
        block::IndexRelations().atMost(loop.getUpperBound(), logicalDimension))
      return true;
  }
  std::optional<int64_t> origin = block::IndexRelations().constant(index);
  return origin &&
         constantOriginInView(*origin, view, resourceAxis, kernel);
}

bool scalarCoordinatesInView(ValueRange coordinates, block::ViewType view,
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
