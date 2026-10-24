#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "mlir/IR/IRMapping.h"
#include <limits>

using namespace mlir;

namespace intent::gpu {

void narrowIntegerArithmetic(func::FuncOp kernel) {
  IntegerRangeAnalysis ranges(integerRangePolicy());
  auto fits = [&](Value value) {
    auto range = ranges.range(value);
    return range && range->smin().sge(std::numeric_limits<int32_t>::min()) &&
           range->smax().sle(std::numeric_limits<int32_t>::max());
  };
  SmallVector<BinaryOp> candidates;
  kernel.walk([&](BinaryOp binary) {
    Type element = uniformElementType(binary.getType());
    auto integer = dyn_cast<IntegerType>(element);
    if (!element.isIndex() && (!integer || integer.isUnsigned() || integer.getWidth() != 64))
      return;
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
    case BinaryOperator::Subtract:
    case BinaryOperator::Multiply:
    case BinaryOperator::FloorDivide:
    case BinaryOperator::Remainder:
    case BinaryOperator::Minimum:
    case BinaryOperator::Maximum:
    case BinaryOperator::BitwiseAnd:
    case BinaryOperator::BitwiseOr:
    case BinaryOperator::BitwiseXor:
      break;
    default: return;
    }
    // Remainder can produce zero even though narrowing would introduce the
    // forbidden signed-min/-1 input pair. Exclude that pair independently.
    if (binary.getOperatorKind() == BinaryOperator::Remainder) {
      auto lhs = ranges.range(binary.getLhs()), rhs = ranges.range(binary.getRhs());
      if (!lhs || !rhs || (lhs->smin().sle(std::numeric_limits<int32_t>::min()) &&
                          rhs->smin().sle(-1) && rhs->smax().sge(-1))) return;
    }
    // No operation may wrap differently at the two bit widths.
    if (fits(binary.getLhs()) && fits(binary.getRhs()) && fits(binary.getResult()))
      candidates.push_back(binary);
  });
  for (BinaryOp binary : candidates) {
    OpBuilder builder(binary);
    auto scalar = builder.getI32Type();
    auto narrowType = [&](Type type) -> Type {
      if (auto fragment = dyn_cast<FragmentType>(type))
        return FragmentType::get(kernel.getContext(), scalar,
            fragment.getShape(), fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      return scalar;
    };
    IRMapping operands;
    for (Value operand : binary->getOperands())
      if (!operands.contains(operand))
        operands.map(operand, builder.create<CastOp>(binary.getLoc(),
            narrowType(operand.getType()), operand).getResult());
    auto replacement = cast<BinaryOp>(builder.clone(*binary, operands));
    replacement.getResult().setType(narrowType(binary.getType()));
    Value result = builder.create<CastOp>(binary.getLoc(), binary.getType(), replacement.getResult());
    binary.replaceAllUsesWith(result);
    binary.erase();
  }
}

} // namespace intent::gpu
