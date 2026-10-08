#include "Intent/Dialect/CPU/Transforms/Vector/VectorReductions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::cpu {

std::optional<IndependentReduction> matchIndependentReduction(
    Value lhs, Value rhs, Value result) {
  if (lhs.getType() != rhs.getType() || lhs.getType() != result.getType())
    return std::nullopt;
  Operation *operation = result.getDefiningOp();
  if (!operation || operation->getNumOperands() != 2 ||
      operation->getNumResults() != 1 ||
      !((operation->getOperand(0) == lhs && operation->getOperand(1) == rhs) ||
        (operation->getOperand(0) == rhs && operation->getOperand(1) == lhs)))
    return std::nullopt;
  Type type = result.getType();
  if (type.isF32() || type.isF64()) {
    if (auto add = dyn_cast<arith::AddFOp>(operation))
      return IndependentReduction{vector::CombiningKind::ADD, add.getFastmath()};
    if (auto multiply = dyn_cast<arith::MulFOp>(operation))
      return IndependentReduction{vector::CombiningKind::MUL, multiply.getFastmath()};
    if (auto maximum = dyn_cast<arith::MaximumFOp>(operation))
      return IndependentReduction{vector::CombiningKind::MAXIMUMF, maximum.getFastmath()};
    if (auto minimum = dyn_cast<arith::MinimumFOp>(operation))
      return IndependentReduction{vector::CombiningKind::MINIMUMF, minimum.getFastmath()};
  } else if (type.isSignlessInteger(8) || type.isSignlessInteger(16) ||
             type.isSignlessInteger(32) || type.isSignlessInteger(64)) {
    if (isa<arith::AddIOp>(operation))
      return IndependentReduction{vector::CombiningKind::ADD, arith::FastMathFlags::none};
    if (isa<arith::MulIOp>(operation))
      return IndependentReduction{vector::CombiningKind::MUL, arith::FastMathFlags::none};
  }
  return std::nullopt;
}

SmallVector<Value> horizontalReduce(
    OpBuilder &builder, Location location, ValueRange values,
    llvm::function_ref<SmallVector<Value>(ValueRange, ValueRange, int64_t)> combine,
    ArrayRef<IndependentReduction> independent) {
  assert(!values.empty());
  auto type = cast<VectorType>(values.front().getType());
  int64_t width = type.getDimSize(0);
  assert(type.getRank() == 1 && !type.isScalable() && llvm::isPowerOf2_64(width));
  for (Value value : values) {
    auto component = cast<VectorType>(value.getType());
    assert(component.getShape() == type.getShape() && !component.isScalable());
    (void)component;
  }
  if (!independent.empty() && width > 1) {
    assert(independent.size() == values.size());
    SmallVector<Value> result;
    for (auto [value, reduction] : llvm::zip(values, independent)) {
      Type element = cast<VectorType>(value.getType()).getElementType();
      auto flags = reduction.fastMath;
      if (isa<FloatType>(element) && (reduction.kind == vector::CombiningKind::ADD ||
                                     reduction.kind == vector::CombiningKind::MUL))
        flags = flags | arith::FastMathFlags::reassoc;
      Value neutral;
      if (isa<FloatType>(element) && reduction.kind == vector::CombiningKind::ADD)
        neutral = builder.create<arith::ConstantOp>(
            location, builder.getFloatAttr(element, -0.0));
      result.push_back(builder.create<vector::ReductionOp>(
          location, reduction.kind, value, neutral, flags));
    }
    return result;
  }
  SmallVector<Value> partial(values);
  for (int64_t count = width; count > 1; count /= 2) {
    SmallVector<int64_t> even, odd;
    for (int64_t lane = 0; lane < count; lane += 2) {
      even.push_back(lane);
      odd.push_back(lane + 1);
    }
    SmallVector<Value> left, right;
    for (Value value : partial) {
      left.push_back(builder.create<vector::ShuffleOp>(location, value, value, even));
      right.push_back(builder.create<vector::ShuffleOp>(location, value, value, odd));
    }
    partial = combine(left, right, count / 2);
  }
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  for (Value &value : partial)
    value = builder.create<vector::ExtractElementOp>(location, value, zero);
  return partial;
}

} // namespace intent::cpu
