#include "IndexBounds.h"
#include "PhysicalProgramDetail.h"
#include "ScalarExpressions.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ConfigurationExpressions.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu {
using namespace detail;

namespace {

Value coordinateSource(Value value) {
  while (true) {
    Value original = value;
    value = stripIntegerIndexCasts(value);
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      value = reshape.getValue();
    else if (auto transpose = value.getDefiningOp<TransposeOp>())
      value = transpose.getValue();
    if (value == original) return value;
  }
}

bool isIndexCoordinate(Value value) {
  if (!value) return false;
  Type type = value.getType();
  if (auto fragment = dyn_cast<FragmentType>(type)) type = fragment.getElementType();
  return type.isIndex();
}

Value completeTileLimit(MakeRangeOp range) {
  auto type = cast<FragmentType>(range.getResult().getType());
  if (!type.getElementType().isIndex() || type.getShape().size() != 1 ||
      integerConstant(range.getStep()) != 1) return {};
  IndexRelations relations;
  for (Operation *owner = range->getParentOp(); owner; owner = owner->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(owner);
    if (!loop || !relations.same(range.getStart(), loop.getInductionVar()) ||
        integerConstant(loop.getLowerBound()) != 0 || !relations.positive(loop.getStep()))
      continue;
    auto width = queryLaunchExpression(loop.getStep());
    if (!width || type.getShape()[0] != width ||
        !relations.same(range.getExtent(), loop.getStep())) continue;
    Value limit = relations.alignedBound(loop.getUpperBound(), loop.getStep());
    // The complete aligned iteration contains precisely [iv, iv + width).
    // Both its lower bound and its representable exclusive endpoint follow
    // together; independent interval arithmetic need not rediscover that fact.
    if (limit && relations.nonnegative(limit)) return limit;
  }
  return {};
}

Value ordinalExtent(Value coordinate) {
  auto subtract = coordinate.getDefiningOp<BinaryOp>();
  if (!subtract || subtract.getOperatorKind() != BinaryOperator::Subtract)
    return {};
  auto range = coordinateSource(subtract.getLhs()).getDefiningOp<MakeRangeOp>();
  if (!range || integerConstant(range.getStep()) != 1 ||
      !sameScalarExpression(stripBroadcast(subtract.getRhs()), range.getStart()) ||
      !IndexRelations().nonnegative(range.getExtent())) return {};
  // (start + lane) - start preserves the lane even if the intermediate wraps.
  return range.getExtent();
}

bool isRangeEndpoint(MakeRangeOp range, Value value) {
  auto add = stripScalarIdentity(value).getDefiningOp<BinaryOp>();
  if (!add || add.getOperatorKind() != BinaryOperator::Add ||
      !value.getType().isIndex() || integerConstant(range.getStep()) != 1 ||
      !IndexRelations().nonnegative(range.getExtent())) return false;
  auto same = [&](Value lhs, Value rhs) { return IndexRelations().same(lhs, rhs); };
  bool matches = (same(add.getLhs(), range.getStart()) && same(add.getRhs(), range.getExtent())) ||
                 (same(add.getRhs(), range.getStart()) && same(add.getLhs(), range.getExtent()));
  return matches && (integerOperationDoesNotWrap(add.getResult()) || completeTileLimit(range));
}

func::FuncOp containingKernel(Value value) {
  return value && value.getParentRegion()
      ? value.getParentRegion()->getParentOfType<func::FuncOp>() : func::FuncOp();
}

bool valueKnownPositive(Value value, unsigned depth);

bool valueKnownNonNegative(Value value, unsigned depth);

bool valueBelowDelinearizeExtent(Value value, Value extent, unsigned depth) {
  if (!value || depth >= 32)
    return false;
  if (auto mapping = queryDecodedCoordinate(value))
      return mapping->extents[mapping->axis] == extent &&
             llvm::all_of(mapping->extents, [&](Value bound) {
               return valueKnownNonNegative(bound, depth + 1);
             });
  Value dividend = IndexRelations().roundedDownSource(value);
  return dividend && valueBelowDelinearizeExtent(dividend, extent, depth + 1);
}

bool valueKnownPositive(Value value, unsigned depth = 0) {
  if (!value || depth >= 32)
    return false;
  value = stripIntegerIndexCasts(value);
  IntegerRangeAnalysis ranges(integerRangePolicy());
  if (ranges.isPositive(value)) return true;
  auto binary = value.getDefiningOp<BinaryOp>();
  if (!binary)
    return false;
  if (binary.getOperatorKind() == BinaryOperator::Subtract)
    return ranges.isNonNegative(binary.getLhs()) &&
           ranges.isNonNegative(binary.getRhs()) &&
           valueBelowDelinearizeExtent(binary.getRhs(), binary.getLhs(), depth + 1);
  if (binary.getOperatorKind() == BinaryOperator::Maximum &&
      value.getType().isIndex())
    return valueKnownPositive(binary.getLhs(), depth + 1) ||
           valueKnownPositive(binary.getRhs(), depth + 1);
  if (binary.getOperatorKind() == BinaryOperator::Minimum ||
      binary.getOperatorKind() == BinaryOperator::MinimumNum)
    return valueKnownPositive(binary.getLhs(), depth + 1) &&
           valueKnownPositive(binary.getRhs(), depth + 1);
  return false;
}

bool valueKnownNonNegative(Value value, unsigned depth = 0) {
  if (!value || depth >= 32)
    return false;
  value = stripIntegerIndexCasts(value);
  IntegerRangeAnalysis ranges(integerRangePolicy());
  if (ranges.isNonNegative(value)) return true;
  if (auto reshape = value.getDefiningOp<ReshapeOp>())
    return valueKnownNonNegative(reshape.getValue(), depth + 1);
  if (auto transpose = value.getDefiningOp<TransposeOp>())
    return valueKnownNonNegative(transpose.getValue(), depth + 1);
  if (auto coordinate = queryDecodedCoordinate(value))
    return llvm::all_of(coordinate->extents, [&](Value extent) {
             return valueKnownNonNegative(extent, depth + 1);
           });
  if (auto coordinate = value.getDefiningOp<WorksetCoordinateOp>())
    return valueKnownNonNegative(coordinate.getCoordinate(), depth + 1);
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<RangeOp>();
    if (!range)
      return false;
    if (bound.getBound() == 0)
      return valueKnownNonNegative(range.getStart(), depth + 1);
    if (bound.getBound() == 1)
      return valueKnownNonNegative(range.getStop(), depth + 1);
    return valueKnownPositive(range.getStep(), depth + 1);
  }
  if (auto select = value.getDefiningOp<SelectOp>())
    return valueKnownNonNegative(select.getTrueValue(), depth + 1) &&
           valueKnownNonNegative(select.getFalseValue(), depth + 1);
  auto binary = value.getDefiningOp<BinaryOp>();
  if (!binary)
    return false;
  bool lhs = valueKnownNonNegative(binary.getLhs(), depth + 1);
  bool rhs = valueKnownNonNegative(binary.getRhs(), depth + 1);
  switch (binary.getOperatorKind()) {
  case BinaryOperator::Minimum:
  case BinaryOperator::MinimumNum:
    return lhs && rhs;
  case BinaryOperator::Maximum:
  case BinaryOperator::MaximumNum:
    return lhs || rhs;
  case BinaryOperator::Subtract: {
    if (lhs && rhs && valueBelowDelinearizeExtent(binary.getRhs(), binary.getLhs(), depth + 1))
      return true;
    if (lhs && rhs) {
      std::function<bool(Value, unsigned)> orderedByCondition =
          [&](Value condition, unsigned remaining) {
        if (remaining == 0)
          return false;
        if (auto conjunction = condition.getDefiningOp<BinaryOp>())
          if (conjunction.getOperatorKind() == BinaryOperator::LogicalAnd)
            return orderedByCondition(conjunction.getLhs(), remaining - 1) ||
                   orderedByCondition(conjunction.getRhs(), remaining - 1);
        auto compare = condition.getDefiningOp<CompareOp>();
        return compare &&
               ((compare.getPredicate() == ComparePredicate::Ge &&
                 compare.getLhs() == binary.getLhs() &&
                 compare.getRhs() == binary.getRhs()) ||
                (compare.getPredicate() == ComparePredicate::Le &&
                 compare.getRhs() == binary.getLhs() &&
                 compare.getLhs() == binary.getRhs()));
      };
      for (Operation *parent = binary->getParentOp(); parent;
           parent = parent->getParentOp())
        if (auto conditional = dyn_cast<scf::IfOp>(parent))
          if (conditional.getThenRegion().isAncestor(binary->getParentRegion()) &&
              orderedByCondition(conditional.getCondition(), 32 - depth))
            return true;
    }
    return false;
  }
  case BinaryOperator::FloorDivide: {
    return lhs && valueKnownPositive(binary.getRhs(), depth + 1);
  }
  case BinaryOperator::Remainder:
    return lhs && valueKnownPositive(binary.getRhs(), depth + 1);
  default:
    return false;
  }
}

Value clampedIncrementLimit(Value value, Value base) {
  auto add = stripScalarIdentity(value).getDefiningOp<BinaryOp>();
  if (!add || !value.getType().isIndex() ||
      add.getOperatorKind() != BinaryOperator::Add)
    return {};
  for (auto [start, increment] :
       {std::pair{add.getLhs(), add.getRhs()},
        std::pair{add.getRhs(), add.getLhs()}}) {
    if (!sameScalarExpression(start, base))
      continue;
    auto minimum = stripScalarIdentity(increment).getDefiningOp<BinaryOp>();
    if (!minimum || minimum.getOperatorKind() != BinaryOperator::Minimum)
      continue;
    for (auto [remaining, step] :
         {std::pair{minimum.getLhs(), minimum.getRhs()},
          std::pair{minimum.getRhs(), minimum.getLhs()}}) {
      auto difference = stripScalarIdentity(remaining).getDefiningOp<BinaryOp>();
      auto amount = integerConstant(step);
      if (difference && difference.getOperatorKind() == BinaryOperator::Subtract &&
          sameScalarExpression(difference.getRhs(), base) && amount && *amount >= 0)
        return difference.getLhs();
    }
  }
  return {};
}

} // namespace

namespace detail {

Value stripIntegerIndexCasts(Value value) {
  value = stripBroadcast(value);
  while (true) {
    if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
      auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
      auto result = dyn_cast<FragmentType>(reshape.getResult().getType());
      auto nonUnitShape = [](ArrayAttr shape) {
        SmallVector<Attribute> extents;
        for (Attribute attribute : shape) {
          auto extent = cast<PhysicalExprAttr>(attribute);
          if (constantPhysicalExpression(extent) != 1)
            extents.push_back(extent);
        }
        return extents;
      };
      // Inserting or removing size-one axes preserves every coordinate value
      // and its linear order, including the coordinate guarded by a mask.
      if (source && result &&
          nonUnitShape(source.getShape()) == nonUnitShape(result.getShape())) {
        value = stripBroadcast(reshape.getValue());
        continue;
      }
    }
    auto cast = value.getDefiningOp<CastOp>();
    if (!cast)
      break;
    auto element = [](Type type) {
      if (auto fragment = dyn_cast<FragmentType>(type))
        return fragment.getElementType();
      return type;
    };
    Type source = element(cast.getValue().getType());
    Type result = element(cast.getResult().getType());
    if (!isa<IntegerType, IndexType>(source) ||
        !isa<IntegerType, IndexType>(result) ||
        !gpu::isValuePreservingIntegerCast(cast.getValue(), result))
      break;
    value = stripBroadcast(cast.getValue());
  }
  return value;
}

bool coordinateKnownNonNegative(Value coordinate) {
  coordinate = coordinateSource(coordinate);
  if (ordinalExtent(coordinate)) return true;
  if (auto range = coordinate.getDefiningOp<MakeRangeOp>();
      range && completeTileLimit(range)) return true;
  if (queryNonNegativeIndexUpperBound(coordinate))
    return true;
  return valueKnownNonNegative(coordinate);
}

bool coordinateKnownPositive(Value coordinate) {
  return valueKnownPositive(coordinate);
}

std::optional<std::pair<int64_t, int64_t>>
nonNegativeExtentBounds(func::FuncOp kernel, PhysicalExprAttr extent) {
  auto bounds = queryPhysicalExpressionRange(extent, kernel);
  if (!bounds || bounds->smin().isNegative()) return std::nullopt;
  return std::pair{bounds->smin().getSExtValue(), bounds->smax().getSExtValue()};
}

std::optional<std::pair<int64_t, int64_t>>
positiveExtentBounds(func::FuncOp kernel, PhysicalExprAttr extent) {
  auto bounds = nonNegativeExtentBounds(kernel, extent);
  return bounds && bounds->first > 0 ? bounds : std::nullopt;
}

IndexBounds queryIndexBounds(Value value) {
  using Bounds = IndexBounds;
  auto constantUpper = [](Bounds bounds) -> std::optional<int64_t> {
    if (bounds.nonNegative && bounds.upper &&
        bounds.upper.getKind() == PhysicalExprKind::Constant)
      return bounds.upper.getValue();
    return std::nullopt;
  };
  std::function<Bounds(Value, unsigned)> bound =
      [&](Value current, unsigned depth) -> Bounds {
    if (!current || depth >= 32 || !current.getType().isIndex())
      return {};
    current = stripScalarIdentity(current);
    auto expression = [&](PhysicalExprKind kind, int64_t constant,
                          ArrayRef<Attribute> operands = {}) {
      MLIRContext *context = current.getContext();
      return PhysicalExprAttr::get(
          context, kind, constant,
          StringAttr::get(context, ""), ArrayAttr::get(context, operands));
    };
    auto numericBounds = [&]() -> Bounds {
      auto range = queryIntegerRange(current);
      if (!range || range->smin().isNegative()) return {};
      int64_t maximum = range->smax().getSExtValue();
      // The index type's limit is not a selected resource capacity.
      PhysicalExprAttr upper = maximum == std::numeric_limits<int64_t>::max()
          ? PhysicalExprAttr() : expression(PhysicalExprKind::Constant, maximum);
      return {true, upper, range->smin().getSExtValue()};
    };
    if (auto coordinate = current.getDefiningOp<WorksetCoordinateOp>())
      return bound(coordinate.getCoordinate(), depth + 1);
    if (current.getDefiningOp<ProgramIdOp>())
      return {true, {}};
    auto dimensionExpression = [&](BlockArgument argument) -> PhysicalExprAttr {
      auto binding = getArgumentBinding(argument);
      return binding && binding.getKind() == ArgumentKind::Dimension
          ? queryArgumentExpression(argument) : PhysicalExprAttr{};
    };
    if (auto dim = current.getDefiningOp<DimOp>())
      return {true, resourceExtentExpression(dim.getView(), dim.getAxis())};
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      if (PhysicalExprAttr dimension = dimensionExpression(argument))
        return {true, dimension};
      auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      if (loop && argument == loop.getInductionVar()) {
        Bounds lower = bound(loop.getLowerBound(), depth + 1);
        Bounds upperBound = bound(loop.getUpperBound(), depth + 1);
        std::optional<int64_t> upper =
            constantUpper(upperBound);
        std::optional<int64_t> step = integerConstant(loop.getStep());
        if (auto parameter = queryParameter(loop.getStep())) {
          auto candidates = parameter.getCandidates().asArrayRef();
          if (parameter.getRole() !=
                  ParameterRole::ResidentWorkers &&
              !candidates.empty() && llvm::all_of(candidates, [](int64_t value) {
                return value > 0;
              }))
            step = *llvm::max_element(candidates);
        }
        if (lower.nonNegative && upper && step && *step > 0 &&
            *upper <= std::numeric_limits<int64_t>::max() - *step + 1)
          return {true, expression(PhysicalExprKind::Constant,
                                   std::max<int64_t>(*upper - 1, 0)), lower.lower};
        if (lower.nonNegative && step && *step == 1) {
          // The induction value exists only in an executing loop body. Its
          // exclusive upper bound therefore exceeds the nonnegative lower
          // bound, even when that upper expression can be negative elsewhere.
          PhysicalExprAttr end = queryLaunchExpression(loop.getUpperBound());
          if (!end && upperBound.nonNegative)
            end = upperBound.upper;
          if (end)
            return {true, expression(PhysicalExprKind::Subtract, 0,
                                    {end, expression(PhysicalExprKind::Constant, 1)}),
                    lower.lower};
        }
      }
      auto whileLoop = dyn_cast<scf::WhileOp>(argument.getOwner()->getParentOp());
      if (whileLoop && argument.getOwner()->getParent() == &whileLoop.getAfter()) {
        OpOperand *forwarded = singleControlInput(
            argument, ControlFlowEdgeKind::RegionTransfer, &whileLoop.getBefore());
        auto condition = forwarded
            ? dyn_cast<scf::ConditionOp>(forwarded->getOwner()) : scf::ConditionOp();
        auto carried = forwarded
            ? dyn_cast<BlockArgument>(stripScalarIdentity(forwarded->get()))
            : BlockArgument();
        auto compare = condition
            ? condition.getCondition().getDefiningOp<CompareOp>() : CompareOp();
        if (carried && carried.getOwner()->getParent() == &whileLoop.getBefore() &&
            compare && compare.getPredicate() == ComparePredicate::Lt &&
            sameScalarExpression(compare.getLhs(), carried)) {
          OpOperand *initialValue = singleControlInput(
              carried, ControlFlowEdgeKind::Entry, nullptr);
          OpOperand *nextValue = singleControlInput(
              carried, ControlFlowEdgeKind::RegionTransfer, &whileLoop.getAfter());
          if (!initialValue || !nextValue) return {};
          Value limit = clampedIncrementLimit(nextValue->get(), argument);
          PhysicalExprAttr end = limit ? queryLaunchExpression(limit) : PhysicalExprAttr();
          Bounds initial = bound(initialValue->get(), depth + 1);
          if (end && queryLaunchExpression(compare.getRhs()) == end &&
              initial.nonNegative && bound(limit, depth + 1).nonNegative) {
            // With 0 <= p < end, p + min(step, end-p) remains in [p,end]
            // without signed overflow. The guard bounds each executing iteration.
            return {true, expression(PhysicalExprKind::Subtract, 0,
                                    {end, expression(PhysicalExprKind::Constant, 1)}),
                    initial.lower};
          }
        }
      }
    }
    if (auto parameter = queryParameter(current)) {
      auto schema = parameter;
      if (llvm::all_of(schema.getCandidates().asArrayRef(),
                       [](int64_t candidate) { return candidate >= 0; }))
        return {true, PhysicalExprAttr::get(
            current.getContext(), PhysicalExprKind::Parameter,
            0, schema.getReference(), ArrayAttr::get(current.getContext(), {}))};
      return {};
    }
    if (auto mapping = queryDecodedCoordinate(current)) {
        if (!llvm::all_of(mapping->extents, [](Value extent) {
              return valueKnownNonNegative(extent);
            }))
          return {};
        Value extent = stripScalarIdentity(
            mapping->extents[mapping->axis]);
        if (std::optional<int64_t> constant = integerConstant(extent)) {
          if (*constant <= 0) return {};
          return {true, expression(PhysicalExprKind::Constant,
                                   *constant - 1)};
        }
        PhysicalExprAttr upper;
        if (auto physical = extent.getDefiningOp<PhysicalExprOp>())
          upper = physical.getExpression();
        else if (auto dim = extent.getDefiningOp<DimOp>())
          upper = resourceExtentExpression(dim.getView(), dim.getAxis());
        else if (auto argument = dyn_cast<BlockArgument>(extent))
          upper = dimensionExpression(argument);
        if (!upper)
          return {};
        return {true, expression(PhysicalExprKind::Maximum, 0,
            {expression(PhysicalExprKind::Subtract, 0,
                        {upper, expression(PhysicalExprKind::Constant, 1)}),
             expression(PhysicalExprKind::Constant, 0)})};
      }
    if (std::optional<int64_t> constant = integerConstant(current)) {
      if (*constant >= 0)
        return {true, expression(PhysicalExprKind::Constant, *constant), *constant};
      return {};
    }
    auto binary = current.getDefiningOp<BinaryOp>();
    if (!binary)
      return numericBounds();
    Bounds lhs = bound(binary.getLhs(), depth + 1);
    Bounds rhs = bound(binary.getRhs(), depth + 1);
    std::optional<int64_t> lhsConstant = constantUpper(lhs);
    std::optional<int64_t> rhsConstant = constantUpper(rhs);
    if (binary.getOperatorKind() == BinaryOperator::Multiply &&
        ((lhsConstant && *lhsConstant == 0) ||
         (rhsConstant && *rhsConstant == 0)))
      return {true, expression(PhysicalExprKind::Constant, 0)};
    if (binary.getOperatorKind() == BinaryOperator::Subtract) {
      PhysicalExprAttr exactLhs = queryLaunchExpression(binary.getLhs());
      if (lhs.nonNegative && rhs.nonNegative && exactLhs && rhs.upper) {
        auto preceding = expression(PhysicalExprKind::Subtract, 0,
                                    {exactLhs, expression(PhysicalExprKind::Constant, 1)});
        if (rhs.upper == exactLhs || rhs.upper == preceding)
          return {true, exactLhs, rhs.upper == preceding ? 1 : 0};
      }
      if (auto constant = integerConstant(binary.getLhs());
          constant && *constant >= 0 && rhsConstant &&
          *rhsConstant <= *constant)
        return {true, expression(PhysicalExprKind::Constant,
                                 *constant - rhs.lower),
                *constant - *rhsConstant};
      auto ordinal = dyn_cast<BlockArgument>(stripScalarIdentity(binary.getRhs()));
      auto loop = ordinal
                      ? dyn_cast<scf::ForOp>(ordinal.getOwner()->getParentOp())
                      : scf::ForOp();
      // Within 0 <= iv < end, end - iv lies in [1, end].  This
      // also proves reversed ordinals without assuming a static extent.
      if (loop && ordinal == loop.getInductionVar() &&
          integerConstant(loop.getStep()) == 1 &&
          bound(loop.getLowerBound(), depth + 1).nonNegative &&
          bound(loop.getUpperBound(), depth + 1).nonNegative) {
        if (sameScalarExpression(binary.getLhs(), loop.getUpperBound()) &&
            lhs.upper)
          return {true, lhs.upper, 1};
        auto last = binary.getLhs().getDefiningOp<BinaryOp>();
        Bounds end = bound(loop.getUpperBound(), depth + 1);
        if (last && last.getOperatorKind() == BinaryOperator::Subtract &&
            integerConstant(last.getRhs()) == 1 && end.upper &&
            sameScalarExpression(last.getLhs(), loop.getUpperBound()))
          return {true, expression(PhysicalExprKind::Subtract, 0,
                                   {end.upper,
                                    expression(PhysicalExprKind::Constant, 1)})};
      }
      std::optional<int64_t> amount = integerConstant(binary.getRhs());
      if (amount && *amount >= 0 && lhs.nonNegative && lhs.upper &&
          lhs.lower >= *amount)
        return {true, expression(PhysicalExprKind::Subtract, 0,
                                 {lhs.upper, expression(PhysicalExprKind::Constant,
                                                        *amount)}),
                lhs.lower - *amount};
    }
    if (binary.getOperatorKind() == BinaryOperator::Add) {
      for (Value start : {binary.getLhs(), binary.getRhs()}) {
        Value limit = clampedIncrementLimit(current, start);
        if (!limit)
          continue;
        Bounds lower = bound(start, depth + 1);
        Bounds upper = bound(limit, depth + 1);
        PhysicalExprAttr end = queryLaunchExpression(limit);
        if (lower.nonNegative && upper.nonNegative && end)
          return {true, end, std::min(lower.lower, upper.lower)};
      }
      for (auto [difference, increment] :
           {std::pair{binary.getLhs(), binary.getRhs()},
            std::pair{binary.getRhs(), binary.getLhs()}}) {
        auto subtract = difference.getDefiningOp<BinaryOp>();
        if (subtract && subtract.getOperatorKind() == BinaryOperator::Subtract &&
            sameScalarExpression(subtract.getRhs(), increment) &&
            valueKnownNonNegative(increment)) {
          // For nonnegative index values E and x, E-x is representable and
          // (E-x)+x is exactly E, including when the intermediate is negative.
          Bounds original = bound(subtract.getLhs(), depth + 1);
          if (original.nonNegative)
            return original;
        }
        std::optional<int64_t> amount = integerConstant(increment);
        Bounds preceding = bound(difference, depth + 1);
        if (amount && *amount >= 0 && preceding.nonNegative && preceding.upper &&
            preceding.upper.getKind() ==
                PhysicalExprKind::Subtract &&
            preceding.upper.getOperands().size() == 2) {
          auto headroom = cast<PhysicalExprAttr>(preceding.upper.getOperands()[1]);
          if (headroom.getKind() ==
                  PhysicalExprKind::Constant &&
              headroom.getValue() >= *amount &&
              preceding.lower <= std::numeric_limits<int64_t>::max() - *amount) {
            auto base = cast<PhysicalExprAttr>(preceding.upper.getOperands()[0]);
            int64_t remaining = headroom.getValue() - *amount;
            return {true, remaining == 0 ? base :
                expression(PhysicalExprKind::Subtract, 0,
                           {base, expression(PhysicalExprKind::Constant, remaining)}),
                    preceding.lower + *amount};
          }
        }
        if (subtract && subtract.getOperatorKind() == BinaryOperator::Subtract &&
            amount && *amount >= 0 &&
            integerConstant(subtract.getRhs()) == amount &&
            preceding.nonNegative) {
          Bounds original = bound(subtract.getLhs(), depth + 1);
          if (original.nonNegative)
            return original;
        }
      }
    }
    if (lhsConstant && rhsConstant &&
        (binary.getOperatorKind() == BinaryOperator::Add ||
         binary.getOperatorKind() == BinaryOperator::Multiply))
      return numericBounds();
    if (binary.getOperatorKind() == BinaryOperator::Add &&
        lhs.nonNegative && rhs.nonNegative && lhs.upper && rhs.upper) {
      PhysicalExprAttr upper = expression(
          PhysicalExprKind::Add, 0, {lhs.upper, rhs.upper});
      auto kernel = binary->getParentOfType<func::FuncOp>();
      if (nonNegativeExtentBounds(kernel, upper))
        return {true, upper, lhs.lower + rhs.lower};
    }
    if (binary.getOperatorKind() == BinaryOperator::Minimum) {
      PhysicalExprAttr upper = lhs.upper && rhs.upper
          ? expression(PhysicalExprKind::Minimum, 0, {lhs.upper, rhs.upper})
          : lhs.upper ? lhs.upper : rhs.upper;
      return {lhs.nonNegative && rhs.nonNegative, upper};
    }
    if (binary.getOperatorKind() == BinaryOperator::Maximum)
      return {lhs.nonNegative || rhs.nonNegative,
              lhs.upper && rhs.upper
                  ? expression(PhysicalExprKind::Maximum, 0, {lhs.upper, rhs.upper})
                  : PhysicalExprAttr()};
    auto positiveDivisor = [&](Value divisor) -> PhysicalExprAttr {
      if (auto constant = integerConstant(divisor))
        return *constant > 0
                   ? expression(PhysicalExprKind::Constant, *constant)
                   : PhysicalExprAttr();
      if (auto parameter = queryParameter(divisor))
        if (llvm::all_of(parameter.getCandidates().asArrayRef(),
                         [](int64_t candidate) { return candidate > 0; }))
          return bound(divisor, depth + 1).upper;
      return {};
    };
    if (binary.getOperatorKind() == BinaryOperator::FloorDivide ||
        binary.getOperatorKind() == BinaryOperator::RightShift) {
      PhysicalExprAttr divisor;
      if (binary.getOperatorKind() == BinaryOperator::FloorDivide) {
        divisor = positiveDivisor(binary.getRhs());
      } else if (auto shift = integerConstant(binary.getRhs());
                 shift && *shift >= 0 && *shift < 63) {
        // Power-of-two division canonicalization must preserve bounds used by
        // native access forms. For a nonnegative index, arithmetic right shift
        // has exactly the same inclusive upper bound as floor division.
        divisor = expression(PhysicalExprKind::Constant, int64_t{1} << *shift);
      }
      return {lhs.nonNegative && bool(divisor),
              lhs.upper && divisor
                  ? expression(PhysicalExprKind::FloorDiv, 0, {lhs.upper, divisor})
                  : PhysicalExprAttr()};
    }
    if (Value dividend = IndexRelations().roundedDownSource(current))
      return bound(dividend, depth + 1);
    return numericBounds();
  };
  return bound(value, 0);
}

} // namespace detail

bool IndexRelations::coordinateLessThan(Value coordinate, Value limit) const {
  if (!isIndexCoordinate(coordinate) || !limit) return false;
  coordinate = coordinateSource(coordinate);
  limit = stripScalarIdentity(limit);
  if (!limit.getType().isIndex()) return false;
  if (coordinate.getType().isIndex() && lessThan(coordinate, limit)) return true;
  auto left = queryIntegerRange(coordinate), right = queryIntegerRange(limit);
  if (left && right && left->smax().slt(right->smin())) return true;
  if (Value extent = ordinalExtent(coordinate))
    if (atMost(extent, limit)) return true;
  if (auto range = coordinate.getDefiningOp<MakeRangeOp>();
      range && integerConstant(range.getStep()) == 1) {
    if (Value end = completeTileLimit(range); end && atMost(end, limit)) return true;
    if (integerConstant(range.getStart()) == 0 && nonnegative(range.getExtent()) &&
        atMost(range.getExtent(), limit)) return true;
    if (isRangeEndpoint(range, limit)) return true;
  }
  auto expression = queryLaunchExpression(limit);
  return expression && coordinateLessThan(coordinate, expression);
}

bool IndexRelations::coordinateLessThan(Value coordinate, PhysicalExprAttr limit) const {
  if (!isIndexCoordinate(coordinate) || !limit) return false;
  coordinate = coordinateSource(coordinate);
  auto kernel = containingKernel(coordinate);
  if (!kernel) return false;
  if (auto exact = queryLaunchExpression(coordinate);
      exact && configurationExpressionLessThan(kernel, exact, limit)) return true;
  if (Value extent = ordinalExtent(coordinate))
    if (atMost(extent, limit)) return true;
  if (auto range = coordinate.getDefiningOp<MakeRangeOp>();
      range && integerConstant(range.getStep()) == 1) {
    if (Value end = completeTileLimit(range); end && atMost(end, limit)) return true;
    if (integerConstant(range.getStart()) == 0 && nonnegative(range.getExtent()) &&
        atMost(range.getExtent(), limit)) return true;
  }
  auto bound = queryIndexBounds(coordinate);
  if (bound.upper && configurationExpressionLessThan(kernel, bound.upper, limit))
    return true;
  auto left = queryIntegerRange(coordinate);
  auto right = queryPhysicalExpressionRange(limit, kernel);
  return left && right && left->smax().slt(right->smin());
}

bool IndexRelations::coordinateInBounds(Value coordinate, PhysicalExprAttr extent) const {
  return nonnegative(coordinate) && coordinateLessThan(coordinate, extent);
}

bool IndexRelations::hasExactRangeEndpoint(Value coordinate, Value limit) const {
  if (!isIndexCoordinate(coordinate) || !limit) return false;
  auto range = coordinateSource(coordinate).getDefiningOp<MakeRangeOp>();
  return range && isRangeEndpoint(range, limit);
}

bool isKnownPositiveExtent(PhysicalExprAttr extent, func::FuncOp kernel) {
  auto range = queryPhysicalExpressionRange(extent, kernel);
  return range && range->smin().isStrictlyPositive();
}

std::optional<std::pair<int64_t, int64_t>>
queryPositiveExtentBounds(PhysicalExprAttr extent, func::FuncOp kernel) {
  return positiveExtentBounds(kernel, extent);
}

PhysicalExprAttr queryNonNegativeIndexUpperBound(Value value) {
  IndexBounds result = queryIndexBounds(value);
  return result.nonNegative ? result.upper : PhysicalExprAttr();
}

PhysicalExprAttr queryLogicalRangeCapacity(MakeRangeOp range) {
  if (!isUnitStepRange(range))
    return {};
  IndexBounds lower = queryIndexBounds(range.getLogicalStart());
  if (lower.nonNegative)
    if (auto stop = queryNonNegativeIndexUpperBound(range.getLogicalStop())) {
      if (lower.lower == 0)
        return stop;
      auto make = [&](PhysicalExprKind kind, int64_t value,
                      ArrayRef<Attribute> operands = {}) {
        auto context = range.getContext();
        return PhysicalExprAttr::get(context, kind, value,
                                    StringAttr::get(context, ""),
                                    ArrayAttr::get(context, operands));
      };
      auto span = make(PhysicalExprKind::Subtract, 0,
                       {stop, make(PhysicalExprKind::Constant, lower.lower)});
      return make(PhysicalExprKind::Maximum, 0,
                  {span, make(PhysicalExprKind::Constant, 0)});
    }

  auto loopBound = [&](Value endpoint, bool upper) -> PhysicalExprAttr {
    Value coordinate = stripScalarIdentity(endpoint);
    if (auto add = coordinate.getDefiningOp<BinaryOp>();
        add && add.getOperatorKind() == BinaryOperator::Add) {
      if (integerConstant(add.getRhs()) == 1)
        coordinate = stripScalarIdentity(add.getLhs());
      else if (integerConstant(add.getLhs()) == 1)
        coordinate = stripScalarIdentity(add.getRhs());
    }
    auto induction = dyn_cast<BlockArgument>(coordinate);
    auto loop = induction
                    ? dyn_cast<scf::ForOp>(induction.getOwner()->getParentOp())
                    : scf::ForOp();
    if (!loop || induction != loop.getInductionVar() ||
        !loop->isAncestor(range) || integerConstant(loop.getStep()) != 1)
      return {};
    // Both iv and iv+1 lie within [lower, upper] in an executing unit-step loop.
    // The successor cannot overflow because iv < upper <= INDEX_MAX.
    return queryLaunchExpression(upper ? loop.getUpperBound()
                                       : loop.getLowerBound());
  };
  auto stop = queryLaunchExpression(range.getLogicalStop());
  if (!stop)
    stop = loopBound(range.getLogicalStop(), /*upper=*/true);
  auto start = queryLaunchExpression(range.getLogicalStart());
  if (!start)
    start = loopBound(range.getLogicalStart(), /*upper=*/false);
  if (!start || !stop)
    return {};
  auto expression = [&](PhysicalExprKind kind, ArrayRef<Attribute> operands) {
    auto context = range.getContext();
    return PhysicalExprAttr::get(context, kind, 0,
                                 StringAttr::get(context, ""),
                                 ArrayAttr::get(context, operands));
  };
  // This is host capacity arithmetic, not a proof that a device subtraction
  // cannot wrap. Coverage selection rejects spans outside its finite domain.
  auto span = expression(PhysicalExprKind::Subtract, {stop, start});
  auto zero = expression(PhysicalExprKind::Constant, {});
  return expression(PhysicalExprKind::Maximum, {span, zero});
}

} // namespace intent::gpu
