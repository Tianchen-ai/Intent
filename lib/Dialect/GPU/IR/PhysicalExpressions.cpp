#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include <limits>

using namespace mlir;

namespace intent::gpu {

std::optional<int64_t> evaluatePhysicalExpression(
    PhysicalExprAttr expression,
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf) {
  if (!expression)
    return std::nullopt;
  auto kind = expression.getKind();
  if (kind == PhysicalExprKind::Constant)
    return expression.getValue();
  if (kind == PhysicalExprKind::Parameter || kind == PhysicalExprKind::Dimension ||
      kind == PhysicalExprKind::ScalarABI)
    return resolveLeaf(expression);
  llvm::SmallVector<int64_t> operands;
  for (Attribute attribute : expression.getOperands()) {
    auto value = evaluatePhysicalExpression(cast<PhysicalExprAttr>(attribute),
                                            resolveLeaf);
    if (!value)
      return std::nullopt;
    operands.push_back(*value);
  }
  __int128 result;
  if (kind == PhysicalExprKind::NextPowerOfTwo && operands.size() == 1) {
    result = 1;
    while (result < operands[0])
      result *= 2;
  } else if (kind == PhysicalExprKind::Select && operands.size() == 3) {
    result = operands[0] ? operands[1] : operands[2];
  } else if (operands.size() == 2) {
    __int128 lhs = operands[0], rhs = operands[1];
    switch (kind) {
    case PhysicalExprKind::Add: result = lhs + rhs; break;
    case PhysicalExprKind::Subtract: result = lhs - rhs; break;
    case PhysicalExprKind::Multiply: result = lhs * rhs; break;
    case PhysicalExprKind::Minimum: result = std::min(lhs, rhs); break;
    case PhysicalExprKind::Maximum: result = std::max(lhs, rhs); break;
    case PhysicalExprKind::FloorDiv:
      if (!rhs) return std::nullopt;
      result = lhs / rhs - (lhs % rhs != 0 && ((lhs < 0) != (rhs < 0)));
      break;
    case PhysicalExprKind::CeilDiv:
      if (!rhs) return std::nullopt;
      result = lhs / rhs + (lhs % rhs != 0 && ((lhs < 0) == (rhs < 0)));
      break;
    default: return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
  if (result < std::numeric_limits<int64_t>::min() ||
      result > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(result);
}

std::optional<int64_t> constantPhysicalExpression(PhysicalExprAttr expression) {
  return evaluatePhysicalExpression(expression, [](PhysicalExprAttr) {
    return std::optional<int64_t>();
  });
}

} // namespace intent::gpu
