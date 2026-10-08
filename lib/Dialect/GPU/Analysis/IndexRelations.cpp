#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Analysis/IntegerRelations.h"
#include "Intent/Dialect/GPU/Analysis/ConfigurationExpressions.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"
#include "IndexBounds.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isCoverageParameter(ParameterAttr parameter) {
  return parameter.isDeferred() ||
         parameter.getCategory() ==
             ParameterCategory::Coverage;
}

bool hasLateBoundDomain(ParameterAttr parameter) {
  // Provider configuration formation binds resident capacity after native
  // access formation. Its placeholder candidates prove neither a constant nor
  // common divisibility of the eventual SM/CTA/occupancy-dependent domain.
  return parameter.getRole() ==
         ParameterRole::ResidentWorkers;
}

std::optional<int64_t> evaluateSingletonExpression(PhysicalExprAttr expression,
                                                  func::FuncOp kernel) {
  return evaluatePhysicalExpression(expression, [&](PhysicalExprAttr leaf)
      -> std::optional<int64_t> {
    if (!kernel || leaf.getKind() !=
                       PhysicalExprKind::Parameter)
      return std::nullopt;
    auto parameter = queryParameterBySymbol(kernel, leaf.getParameterReference().getName());
    if (failed(parameter) || isCoverageParameter(*parameter) ||
        hasLateBoundDomain(*parameter))
      return std::nullopt;
    auto candidates = parameter->getCandidates().asArrayRef();
    return candidates.size() == 1 ? std::optional<int64_t>(candidates.front())
                                  : std::nullopt;
  });
}

} // namespace

std::optional<int64_t> IndexRelations::constant(PhysicalExprAttr expression,
                                                func::FuncOp kernel) const {
  return evaluateSingletonExpression(expression, kernel);
}

std::optional<int64_t> IndexRelations::constant(Value value) const {
  if (!value || !value.getType().isIntOrIndex())
    return std::nullopt;
  if (PhysicalExprAttr expression = queryLaunchExpression(value)) {
    Operation *owner = value.getDefiningOp();
    if (!owner)
      owner = cast<BlockArgument>(value).getOwner()->getParentOp();
    auto kernel = dyn_cast<func::FuncOp>(owner);
    if (!kernel)
      kernel = owner->getParentOfType<func::FuncOp>();
    // Do not retry overflowing launch arithmetic with modular constant folding.
    return evaluateSingletonExpression(expression, kernel);
  }
  auto integer = dyn_cast_or_null<IntegerAttr>(
      UniformValueAnalysis(describeUniformValue).evaluate(value));
  if (!integer || integer.getValue().getBitWidth() > 64)
    return std::nullopt;
  auto type = dyn_cast<IntegerType>(value.getType());
  if (type && (type.isUnsigned() || type.getWidth() == 1))
    return integer.getValue().getActiveBits() <= 63
               ? std::optional<int64_t>(integer.getValue().getZExtValue())
               : std::nullopt;
  return integer.getInt();
}

bool IndexRelations::same(Value lhs, Value rhs) const {
  if (samePhysicalScalarExpression(lhs, rhs))
    return true;
  auto left = queryLaunchExpression(lhs), right = queryLaunchExpression(rhs);
  return left && right && left == right;
}

bool IndexRelations::nonnegative(Value value) const {
  Type type = value ? value.getType() : Type();
  if (auto fragment = dyn_cast_or_null<FragmentType>(type)) type = fragment.getElementType();
  if (!type || !type.isIndex())
    return false;
  if (auto range = queryIntegerRange(value); range && range->smin().isNonNegative())
    return true;
  // Coordinate/control relations can prove facts that independent value
  // intervals cannot: e.g. extent - coordinate within its decoded domain.
  return detail::coordinateKnownNonNegative(value);
}

bool IndexRelations::positive(Value value) const {
  if (!value || !value.getType().isIndex()) return false;
  if (auto range = queryIntegerRange(value); range && range->smin().isStrictlyPositive())
    return true;
  auto expression = queryLaunchExpression(value);
  auto owner = value.getDefiningOp();
  auto kernel = owner ? owner->getParentOfType<func::FuncOp>() : func::FuncOp();
  return (expression && kernel && isKnownPositiveExtent(expression, kernel)) ||
         detail::coordinateKnownPositive(value);
}

namespace {

PhysicalExprAttr physical(OpFoldResult node) {
  return dyn_cast_or_null<PhysicalExprAttr>(dyn_cast<Attribute>(node));
}

IntegerOrder<OpFoldResult> indexOrder(const IndexRelations &relations,
                                    func::FuncOp kernel) {
  IntegerOrderCallbacks<OpFoldResult> callbacks;
  callbacks.signedWidth = [](OpFoldResult node) -> unsigned {
    if (auto value = dyn_cast<Value>(node)) return value.getType().isIndex() ? 64 : 0;
    return physical(node) ? 64 : 0;
  };
  callbacks.binary = [](OpFoldResult node) -> std::optional<IntegerOrderBinary<OpFoldResult>> {
    IntegerOrderKind kind;
    if (auto value = dyn_cast<Value>(node)) {
      auto binary = value.getDefiningOp<BinaryOp>();
      if (!binary) return std::nullopt;
      switch (binary.getOperatorKind()) {
      case BinaryOperator::Add: kind = IntegerOrderKind::Add; break;
      case BinaryOperator::Subtract: kind = IntegerOrderKind::Subtract; break;
      case BinaryOperator::Multiply: kind = IntegerOrderKind::Multiply; break;
      case BinaryOperator::FloorDivide: kind = IntegerOrderKind::FloorDivide; break;
      case BinaryOperator::Minimum: kind = IntegerOrderKind::Minimum; break;
      case BinaryOperator::Maximum: kind = IntegerOrderKind::Maximum; break;
      default: return std::nullopt;
      }
      return IntegerOrderBinary<OpFoldResult>{kind, binary.getLhs(), binary.getRhs()};
    }
    auto expression = physical(node);
    if (!expression || expression.getOperands().size() != 2) return std::nullopt;
    switch (expression.getKind()) {
    case PhysicalExprKind::Add: kind = IntegerOrderKind::Add; break;
    case PhysicalExprKind::Subtract: kind = IntegerOrderKind::Subtract; break;
    case PhysicalExprKind::Multiply: kind = IntegerOrderKind::Multiply; break;
    case PhysicalExprKind::FloorDiv: kind = IntegerOrderKind::FloorDivide; break;
    case PhysicalExprKind::Minimum: kind = IntegerOrderKind::Minimum; break;
    case PhysicalExprKind::Maximum: kind = IntegerOrderKind::Maximum; break;
    default: return std::nullopt;
    }
    return IntegerOrderBinary<OpFoldResult>{kind, expression.getOperands()[0], expression.getOperands()[1]};
  };
  callbacks.constant = [&relations, kernel](OpFoldResult node) {
    if (auto value = dyn_cast<Value>(node)) return relations.constant(value);
    return relations.constant(physical(node), kernel);
  };
  auto expression = [](OpFoldResult node) {
    if (auto value = dyn_cast<Value>(node)) return queryLaunchExpression(value);
    return physical(node);
  };
  callbacks.same = [&relations, expression](OpFoldResult lhs, OpFoldResult rhs) {
    auto left = dyn_cast<Value>(lhs), right = dyn_cast<Value>(rhs);
    if (left && right) return relations.same(left, right);
    auto a = expression(lhs), b = expression(rhs);
    return a && b && a == b;
  };
  callbacks.nonnegative = [&relations, kernel](OpFoldResult node) {
    if (auto value = dyn_cast<Value>(node)) return relations.nonnegative(value);
    auto range = queryPhysicalExpressionRange(physical(node), kernel);
    return range && range->smin().isNonNegative();
  };
  callbacks.positive = [&relations, kernel](OpFoldResult node) {
    if (auto value = dyn_cast<Value>(node)) return relations.positive(value);
    auto range = queryPhysicalExpressionRange(physical(node), kernel);
    return range && range->smin().isStrictlyPositive();
  };
  callbacks.noSignedWrap = [kernel](OpFoldResult node) {
    if (auto value = dyn_cast<Value>(node)) return integerOperationDoesNotWrap(value);
    return bool(queryPhysicalExpressionRange(physical(node), kernel));
  };
  callbacks.knownOrder = [kernel, expression](OpFoldResult lhs, OpFoldResult rhs, bool strict) {
    auto left = dyn_cast<Value>(lhs), right = dyn_cast<Value>(rhs);
    auto a = left ? queryIntegerRange(left) : queryPhysicalExpressionRange(physical(lhs), kernel);
    auto b = right ? queryIntegerRange(right) : queryPhysicalExpressionRange(physical(rhs), kernel);
    if (a && b && (strict ? a->smax().slt(b->smin()) : a->smax().sle(b->smin()))) return true;
    auto first = expression(lhs), second = expression(rhs);
    return first && second && (strict ? configurationExpressionLessThan(kernel, first, second)
                                     : configurationExpressionAtMost(kernel, first, second));
  };
  return IntegerOrder<OpFoldResult>(std::move(callbacks));
}

func::FuncOp kernelOf(Value value) {
  return value && value.getParentRegion()
      ? value.getParentRegion()->getParentOfType<func::FuncOp>() : func::FuncOp();
}

} // namespace

bool IndexRelations::atMost(Value lhs, Value rhs) const {
  if (!lhs || !rhs) return false;
  return indexOrder(*this, kernelOf(lhs)).atMost(lhs, rhs);
}

bool IndexRelations::lessThan(Value lhs, Value rhs) const {
  if (!lhs || !rhs) return false;
  return indexOrder(*this, kernelOf(lhs)).lessThan(lhs, rhs);
}

bool IndexRelations::atMost(Value lhs, PhysicalExprAttr rhs) const {
  if (!lhs || !rhs) return false;
  if (indexOrder(*this, kernelOf(lhs)).atMost(lhs, rhs)) return true;
  auto upper = queryNonNegativeIndexUpperBound(lhs);
  return upper && configurationExpressionAtMost(kernelOf(lhs), upper, rhs);
}

Value IndexRelations::roundedDownSource(Value value) const {
  if (!value) return {};
  auto rounded = indexOrder(*this, kernelOf(value)).roundDown(value);
  return rounded ? dyn_cast<Value>(rounded->dividend) : Value();
}

bool IndexRelations::powerOfTwo(Value value) const {
  if (auto parameter = queryParameter(value)) {
    if (hasLateBoundDomain(parameter))
      return false;
    auto candidates = parameter.getCandidates().asArrayRef();
    return !candidates.empty() && llvm::all_of(candidates, [](int64_t candidate) {
      return candidate > 0 && llvm::isPowerOf2_64(candidate);
    });
  }
  auto literal = constant(value);
  return literal && *literal > 0 && llvm::isPowerOf2_64(*literal);
}

bool IndexRelations::multipleOf(
    Value value, int64_t divisor,
    llvm::function_ref<bool(ParameterAttr)> alignedParameter) const {
  if (!value || !value.getType().isIndex() || divisor <= 0)
    return false;
  // Powers of two divide the index modulus. Their divisibility survives
  // fixed-width add/sub/mul; other divisors require a concrete constant.
  auto declarationAligned = [&](ParameterAttr parameter) {
    if (alignedParameter && alignedParameter(parameter))
      return true;
    if (hasLateBoundDomain(parameter))
      return false;
    auto candidates = parameter.getCandidates().asArrayRef();
    return !candidates.empty() && llvm::all_of(candidates, [&](int64_t candidate) {
      return candidate % divisor == 0;
    });
  };
  std::function<bool(Value, unsigned)> prove = [&](Value current, unsigned depth) {
    if (!current || !current.getType().isIndex() || depth >= 32)
      return false;
    if (divisor == 1)
      return true;
    if (auto parameter = queryParameter(current))
      return declarationAligned(parameter);
    if (auto literal = constant(current))
      return *literal % divisor == 0;
    if (!llvm::isPowerOf2_64(divisor))
      return false;
    auto aligned = [&](Value operand) { return prove(operand, depth + 1); };
    if (auto coordinate = current.getDefiningOp<WorksetCoordinateOp>())
      return aligned(coordinate.getCoordinate());
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      return loop && argument == loop.getInductionVar() &&
             aligned(loop.getLowerBound()) && aligned(loop.getStep());
    }
    if (auto select = current.getDefiningOp<SelectOp>())
      return aligned(select.getTrueValue()) && aligned(select.getFalseValue());
    if (auto physical = current.getDefiningOp<PhysicalExprOp>()) {
      std::function<bool(PhysicalExprAttr)> expressionAligned =
          [&](PhysicalExprAttr expression) {
        auto kind = expression.getKind();
        if (kind == PhysicalExprKind::Constant)
          return expression.getValue() % divisor == 0;
        if (kind == PhysicalExprKind::Parameter) {
          auto parameter = queryParameterBySymbol(
              physical->getParentOfType<func::FuncOp>(), expression.getParameterReference().getName());
          return succeeded(parameter) && declarationAligned(*parameter);
        }
        if (expression.getOperands().size() != 2)
          return false;
        bool lhs = expressionAligned(cast<PhysicalExprAttr>(expression.getOperands()[0]));
        bool rhs = expressionAligned(cast<PhysicalExprAttr>(expression.getOperands()[1]));
        if (kind == PhysicalExprKind::Multiply)
          return lhs || rhs;
        return (kind == PhysicalExprKind::Add || kind == PhysicalExprKind::Subtract ||
                kind == PhysicalExprKind::Minimum || kind == PhysicalExprKind::Maximum) &&
               lhs && rhs;
      };
      return expressionAligned(physical.getExpression());
    }
    if (auto binary = current.getDefiningOp<BinaryOp>()) {
      auto kind = binary.getOperatorKind();
      if (kind == BinaryOperator::Multiply)
        return aligned(binary.getLhs()) || aligned(binary.getRhs());
      if (kind == BinaryOperator::Add || kind == BinaryOperator::Subtract ||
          kind == BinaryOperator::Minimum || kind == BinaryOperator::Maximum)
        return aligned(binary.getLhs()) && aligned(binary.getRhs());
    }
    return false;
  };
  return prove(value, 0);
}

bool IndexRelations::multipleOf(Value value, Value divisor) const {
  if (!value || !divisor || !value.getType().isIndex() ||
      !divisor.getType().isIndex() || !positive(divisor))
    return false;
  if (same(value, divisor) || constant(value) == 0)
    return true;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    // Keep the correlation between an induction variable and its actual step.
    // Enumerating each step candidate independently loses this identity.
    if (loop && argument == loop.getInductionVar() &&
        loop.getStep() == divisor && constant(loop.getLowerBound()) == 0 &&
        powerOfTwo(divisor))
      return true;
  }
  if (auto literal = constant(divisor))
    return multipleOf(value, *literal);
  auto parameter = queryParameter(divisor);
  if (!parameter || hasLateBoundDomain(parameter))
    return false;
  auto candidates = parameter.getCandidates().asArrayRef();
  if (candidates.empty())
    return false;
  std::function<bool(Value, unsigned)> prove = [&](Value current, unsigned depth) {
    if (!current || !current.getType().isIndex() || depth >= 32)
      return false;
    if (same(current, divisor) || constant(current) == 0)
      return true;
    if (llvm::all_of(candidates, [&](int64_t candidate) {
          return multipleOf(current, candidate);
        }))
      return true;
    if (!powerOfTwo(divisor))
      return false;
    auto aligned = [&](Value operand) { return prove(operand, depth + 1); };
    if (auto coordinate = current.getDefiningOp<WorksetCoordinateOp>())
      return aligned(coordinate.getCoordinate());
    if (auto binary = current.getDefiningOp<BinaryOp>()) {
      auto kind = binary.getOperatorKind();
      if (kind == BinaryOperator::Multiply)
        return aligned(binary.getLhs()) || aligned(binary.getRhs());
      if (kind == BinaryOperator::Add || kind == BinaryOperator::Subtract ||
          kind == BinaryOperator::Minimum || kind == BinaryOperator::Maximum)
        return aligned(binary.getLhs()) && aligned(binary.getRhs());
    }
    return false;
  };
  return prove(value, 0);
}

bool IndexRelations::alignedUnitWindow(Value start, Value extent) const {
  if (!start || !extent || !start.getType().isIndex() ||
      !extent.getType().isIndex() || !powerOfTwo(extent) ||
      !multipleOf(start, extent))
    return false;
  // For a represented signed index aligned to 2^k, the largest possible start
  // is INT64_MAX - (2^k - 1). Adding a nonnegative offset below 2^k is safe,
  // including for negative origins and modular arithmetic before this window.
  return true;
}

Value IndexRelations::alignedBound(Value value, Value step) const {
  if (constant(value) == 0 || same(value, step))
    return value;
  auto operation = value.getDefiningOp<BinaryOp>();
  if (!operation)
    return {};
  if (auto rounded = indexOrder(*this, kernelOf(value)).roundDown(value))
    if (Value divisor = dyn_cast<Value>(rounded->divisor); divisor && same(divisor, step))
      return value;
  if (operation.getOperatorKind() == BinaryOperator::Minimum)
    for (auto [candidate, other] :
         {std::pair{operation.getLhs(), operation.getRhs()},
          std::pair{operation.getRhs(), operation.getLhs()}})
      if (Value aligned = alignedBound(candidate, step);
          aligned && atMost(candidate, other))
        return aligned;
  return {};
}

} // namespace intent::gpu
