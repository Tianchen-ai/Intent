#include "Intent/Analysis/IntegerRelations.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;
namespace intent {

IntegerDifference foldIntegerDifference(
    const UniformExpression &expression, IntegerDifferenceQuery difference,
    llvm::function_ref<std::optional<int64_t>(Value)> constant,
    unsigned indexBitWidth) {
  auto width = [&](Type type) -> unsigned {
    if (!type) return 0;
    if (type.isIndex()) return indexBitWidth;
    auto integer = dyn_cast<IntegerType>(type);
    return integer ? integer.getWidth() : 0;
  };
  if (width(expression.type) != 64) return std::nullopt;
  if (expression.kind == UniformKind::Cast) {
    // Equal-width integer/index conversion preserves the bit pattern. A
    // narrowing or widening cast needs a separate no-wrap/range proof.
    if (expression.operands.size() != 1 || width(expression.operands.front().getType()) != 64)
      return std::nullopt;
    return difference(expression.operands.front());
  }
  if (expression.operands.size() != 2 ||
      width(expression.operands[0].getType()) != 64 ||
      width(expression.operands[1].getType()) != 64) return std::nullopt;
  auto checked = [](const llvm::APInt &coefficient) -> IntegerDifference {
    return coefficient.isSignedIntN(64) ? IntegerDifference(coefficient.getSExtValue()) : std::nullopt;
  };
  Value lhs = expression.operands[0], rhs = expression.operands[1];
  if (expression.kind == UniformKind::Add || expression.kind == UniformKind::Subtract) {
    auto left = difference(lhs), right = difference(rhs);
    if (!left || !right) return std::nullopt;
    llvm::APInt result(128, *left, true), other(128, *right, true);
    return checked(expression.kind == UniformKind::Add ? result + other : result - other);
  }
  if (expression.kind == UniformKind::Multiply) {
    auto scale = constant(lhs);
    Value input = rhs;
    if (!scale) { scale = constant(rhs); input = lhs; }
    if (!scale) return std::nullopt;
    auto coefficient = difference(input);
    if (!coefficient) return std::nullopt;
    return checked(llvm::APInt(128, *coefficient, true) * llvm::APInt(128, *scale, true));
  }
  return std::nullopt;
}

} // namespace intent
