#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Analysis/IntegerRelations.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;

namespace intent::gpu {
namespace {

using Range = ConstantIntRanges;

Range signedRange(int64_t lower, int64_t upper) {
  return Range::fromSigned(APInt(64, lower, true), APInt(64, upper, true));
}

std::optional<IntegerOrderBinary<PhysicalExprAttr>>
binaryExpression(PhysicalExprAttr expression) {
  if (!expression || expression.getOperands().size() != 2)
    return std::nullopt;
  IntegerOrderKind kind;
  switch (expression.getKind()) {
  case PhysicalExprKind::Add: kind = IntegerOrderKind::Add; break;
  case PhysicalExprKind::Subtract: kind = IntegerOrderKind::Subtract; break;
  case PhysicalExprKind::Multiply: kind = IntegerOrderKind::Multiply; break;
  case PhysicalExprKind::FloorDiv: kind = IntegerOrderKind::FloorDivide; break;
  case PhysicalExprKind::Minimum: kind = IntegerOrderKind::Minimum; break;
  case PhysicalExprKind::Maximum: kind = IntegerOrderKind::Maximum; break;
  default: return std::nullopt;
  }
  return IntegerOrderBinary<PhysicalExprAttr>{
      kind, cast<PhysicalExprAttr>(expression.getOperands()[0]),
      cast<PhysicalExprAttr>(expression.getOperands()[1])};
}

std::optional<Range> divisionBounds(PhysicalExprAttr expression,
                                    const Range &lhs, const Range &rhs) {
  if (rhs.smin().isNonPositive() && rhs.smax().isNonNegative())
    return std::nullopt;
  auto *context = expression.getContext();
  auto literal = [&](int64_t value) {
    return PhysicalExprAttr::get(context, PhysicalExprKind::Constant, value,
                                StringAttr::get(context, ""),
                                ArrayAttr::get(context, {}));
  };
  int64_t lower = std::numeric_limits<int64_t>::max();
  int64_t upper = std::numeric_limits<int64_t>::min();
  // For a divisor interval not crossing zero, extrema occur at these corners.
  // Use the host expression's checked evaluator: native arith range inference
  // may account for a different constant-folding contract at signed INT_MIN.
  for (const APInt &numerator : {lhs.smin(), lhs.smax()})
    for (const APInt &denominator : {rhs.smin(), rhs.smax()}) {
      auto concrete = PhysicalExprAttr::get(
          context, expression.getKind(), 0, StringAttr::get(context, ""),
          ArrayAttr::get(context, {literal(numerator.getSExtValue()),
                                  literal(denominator.getSExtValue())}));
      auto value = constantPhysicalExpression(concrete);
      if (!value) return std::nullopt;
      lower = std::min(lower, *value);
      upper = std::max(upper, *value);
    }
  return signedRange(lower, upper);
}

class PhysicalExpressionBounds {
public:
  explicit PhysicalExpressionBounds(func::FuncOp kernel) : kernel(kernel) {}

  std::optional<Range> query(PhysicalExprAttr expression) {
    if (!expression)
      return std::nullopt;
    if (auto found = known.find(expression); found != known.end())
      return found->second;
    auto result = infer(expression);
    known.try_emplace(expression, result);
    return result;
  }

  IntegerOrder<PhysicalExprAttr> ordering() {
    IntegerOrderCallbacks<PhysicalExprAttr> callbacks;
    callbacks.signedWidth = [](PhysicalExprAttr expression) {
      return expression ? 64u : 0u;
    };
    callbacks.binary = binaryExpression;
    callbacks.constant = [this](PhysicalExprAttr expression) -> std::optional<int64_t> {
      auto range = query(expression);
      if (range)
        if (auto value = range->getConstantValue()) return value->getSExtValue();
      return std::nullopt;
    };
    callbacks.same = [](PhysicalExprAttr lhs, PhysicalExprAttr rhs) { return lhs == rhs; };
    callbacks.nonnegative = [this](PhysicalExprAttr expression) {
      auto range = query(expression);
      return range && range->smin().isNonNegative();
    };
    callbacks.positive = [this](PhysicalExprAttr expression) {
      auto range = query(expression);
      return range && range->smin().isStrictlyPositive();
    };
    callbacks.noSignedWrap = [this](PhysicalExprAttr expression) {
      return query(expression).has_value();
    };
    callbacks.knownOrder = [this](PhysicalExprAttr lhs, PhysicalExprAttr rhs, bool strict) {
      if (!strict && kernel && rhs.getKind() == PhysicalExprKind::Parameter) {
        auto parameter = lookupParameter(kernel, rhs.getParameterReference());
        if (parameter && parameter.getCategory() == ParameterCategory::Coverage &&
            parameter.getBinding().getCoverageBound() == lhs)
          return true;
      }
      auto left = query(lhs), right = query(rhs);
      return left && right && (strict ? left->smax().slt(right->smin())
                                     : left->smax().sle(right->smin()));
    };
    return IntegerOrder<PhysicalExprAttr>(std::move(callbacks));
  }

private:
  std::optional<Range> parameterRange(PhysicalExprAttr expression) {
    if (!kernel)
      return std::nullopt;
    auto parameter = lookupParameter(kernel, expression.getParameterReference());
    if (!parameter)
      return std::nullopt;
    // Native configuration formation may replace these placeholder candidates.
    // Only the positive capacity contract survives that later binding.
    if (parameter.getRole() == ParameterRole::ResidentWorkers)
      return signedRange(1, std::numeric_limits<int64_t>::max());
    auto candidates = parameter.getCandidates().asArrayRef();
    if (candidates.empty())
      return std::nullopt;
    int64_t lower = *llvm::min_element(candidates);
    int64_t upper = *llvm::max_element(candidates);
    // Deferred coverage also selects an element of its declared finite domain.
    // A runtime extent that no candidate covers is rejected by the binder.
    if (auto configurations = kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
        configurations && !configurations.getRows().empty()) {
      int64_t selectedLower = std::numeric_limits<int64_t>::max();
      int64_t selectedUpper = std::numeric_limits<int64_t>::min();
      bool complete = true;
      for (Attribute attribute : configurations.getRows()) {
        auto row = dyn_cast<DictionaryAttr>(attribute);
        auto selected = row ? row.getAs<IntegerAttr>(parameter.getName()) : IntegerAttr();
        if (!selected || !llvm::is_contained(candidates, selected.getInt())) {
          complete = false;
          break;
        }
        selectedLower = std::min(selectedLower, selected.getInt());
        selectedUpper = std::max(selectedUpper, selected.getInt());
      }
      if (complete) {
        lower = selectedLower;
        upper = selectedUpper;
      }
    }
    return signedRange(lower, upper);
  }

  std::optional<Range> infer(PhysicalExprAttr expression) {
    auto kind = expression.getKind();
    if (kind == PhysicalExprKind::Constant)
      return Range::constant(APInt(64, expression.getValue(), true));
    if (kind == PhysicalExprKind::Dimension)
      return signedRange(0, std::numeric_limits<int64_t>::max());
    if (kind == PhysicalExprKind::ScalarABI)
      return Range::maxRange(64);
    if (kind == PhysicalExprKind::Parameter)
      return parameterRange(expression);
    std::optional<Range> correlated;
    if (auto rounded = ordering().roundDown(expression)) {
      // This relation proves the product itself cannot overflow, unlike the
      // Cartesian product of independently inferred quotient/divisor ranges.
      auto dividend = query(rounded->dividend);
      correlated = signedRange(0, dividend->smax().getSExtValue());
    }
    SmallVector<Range> operands;
    for (Attribute attribute : expression.getOperands()) {
      auto range = query(cast<PhysicalExprAttr>(attribute));
      if (!range)
        return std::nullopt;
      operands.push_back(*range);
    }
    if (kind == PhysicalExprKind::Select && operands.size() == 3) {
      if (auto constant = operands[0].getConstantValue())
        return constant->isZero() ? operands[2] : operands[1];
      return operands[1].rangeUnion(operands[2]);
    }
    if (kind == PhysicalExprKind::NextPowerOfTwo && operands.size() == 1) {
      int64_t lower = operands[0].smin().getSExtValue();
      int64_t upper = operands[0].smax().getSExtValue();
      if (upper > (int64_t{1} << 62))
        return std::nullopt;
      return signedRange(llvm::PowerOf2Ceil(std::max<int64_t>(1, lower)),
                         llvm::PowerOf2Ceil(std::max<int64_t>(1, upper)));
    }
    if (operands.size() != 2)
      return std::nullopt;
    auto binary = [&](BinaryOperator operation) -> std::optional<Range> {
      if (!provesSignedNoWrap(operation, operands[0], operands[1]))
        return std::nullopt;
      return inferIntegerBinary(operation, IndexType::get(expression.getContext()),
                                operands[0], operands[1]);
    };
    switch (kind) {
    case PhysicalExprKind::Add:
      return binary(BinaryOperator::Add);
    case PhysicalExprKind::Subtract:
      return binary(BinaryOperator::Subtract);
    case PhysicalExprKind::Multiply: {
      auto independent = binary(BinaryOperator::Multiply);
      // Correlation supplements ordinary inference, retaining exact constants
      // and positive lower bounds already established by the operand domains.
      if (!correlated) return independent;
      return independent ? independent->intersection(*correlated) : correlated;
    }
    case PhysicalExprKind::Minimum:
      return intrange::inferMinS(operands);
    case PhysicalExprKind::Maximum:
      return intrange::inferMaxS(operands);
    case PhysicalExprKind::FloorDiv:
    case PhysicalExprKind::CeilDiv:
      return divisionBounds(expression, operands[0], operands[1]);
    default:
      return std::nullopt;
    }
  }

  func::FuncOp kernel;
  DenseMap<PhysicalExprAttr, std::optional<Range>> known;
};

} // namespace

std::optional<ConstantIntRanges>
queryPhysicalExpressionRange(PhysicalExprAttr expression, func::FuncOp kernel) {
  return PhysicalExpressionBounds(kernel).query(expression);
}

bool physicalExpressionAtMost(PhysicalExprAttr lhs, PhysicalExprAttr rhs,
                              func::FuncOp kernel) {
  PhysicalExpressionBounds bounds(kernel);
  return bounds.ordering().atMost(lhs, rhs);
}

bool physicalExpressionLessThan(PhysicalExprAttr lhs, PhysicalExprAttr rhs,
                                func::FuncOp kernel) {
  PhysicalExpressionBounds bounds(kernel);
  return bounds.ordering().lessThan(lhs, rhs);
}

} // namespace intent::gpu
