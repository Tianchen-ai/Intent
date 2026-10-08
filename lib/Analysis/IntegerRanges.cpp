#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Analysis/ControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/ScopeExit.h"
#include <algorithm>

using namespace mlir;

namespace intent {
namespace {

unsigned bitWidth(Type type) {
  if (!type) return 0;
  if (type.isIndex()) return 64;
  if (auto integer = dyn_cast<IntegerType>(type)) return integer.getWidth();
  return 0;
}

bool isUnsigned(Type type) {
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.isUnsigned() || integer.getWidth() == 1);
}

ConstantIntRanges nonNegativeDimension() {
  return ConstantIntRanges::fromSigned(APInt(64, 0),
                                      APInt::getSignedMaxValue(64));
}

ConstantIntRanges truthValue(const ConstantIntRanges &input) {
  APInt zero(input.umin().getBitWidth(), 0);
  auto result = intrange::evaluatePred(intrange::CmpPredicate::ne, input,
                                      ConstantIntRanges::constant(zero));
  return result ? ConstantIntRanges::constant(APInt(1, *result))
                : ConstantIntRanges::maxRange(1);
}

// Intent remainder uses floor division, whereas arith.remsi truncates towards
// zero. The divisor sign determines this range, including negative dividends.
ConstantIntRanges floorRemainder(const ConstantIntRanges &lhs,
                                 const ConstantIntRanges &rhs) {
  unsigned width = lhs.smin().getBitWidth();
  auto dividend = lhs.getConstantValue(), divisor = rhs.getConstantValue();
  if (dividend && divisor && !divisor->isZero() &&
      !(dividend->isMinSignedValue() && divisor->isAllOnes())) {
    APInt remainder = dividend->srem(*divisor);
    if (!remainder.isZero() && remainder.isNegative() != divisor->isNegative())
      remainder += *divisor;
    return ConstantIntRanges::constant(remainder);
  }
  APInt zero(width, 0), one(width, 1);
  if (rhs.smin().sgt(zero))
    return ConstantIntRanges::fromSigned(zero, rhs.smax() - one);
  if (rhs.smax().slt(zero))
    return ConstantIntRanges::fromSigned(rhs.smin() + one, zero);
  return ConstantIntRanges::maxRange(width);
}

// Descriptor models own dimension equalities. Stop at integer SSA leaves so
// their potentially wrapping arithmetic is interpreted by InferIntRange.
bool stopAtDimensionLeaf(Value value, std::optional<int64_t> dimension,
                          ValueBoundsConstraintSet &) {
  Operation *owner = value.getDefiningOp();
  if (auto argument = dyn_cast<BlockArgument>(value))
    owner = argument.getOwner()->getParentOp();
  if (dimension) {
    if (!isa_and_nonnull<ValueBoundsOpInterface>(owner)) return true;
    // Construction can query a descriptor before sibling control regions have
    // terminators. Their ValueBounds models may only inspect complete regions.
    return !hasCompleteControlFlowRegions(owner);
  }
  return !isa_and_nonnull<arith::ConstantOp, memref::DimOp, tensor::DimOp,
                         memref::RankOp, tensor::RankOp,
                         memref::ExtractStridedMetadataOp>(owner);
}

std::optional<ConstantIntRanges>
evaluateDimensionExpression(AffineExpr expression,
                            ArrayRef<ConstantIntRanges> operands,
                            unsigned dimensions) {
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return ConstantIntRanges::constant(APInt(64, constant.getValue(), true));
  if (auto dimension = dyn_cast<AffineDimExpr>(expression))
    return operands[dimension.getPosition()];
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expression))
    return operands[dimensions + symbol.getPosition()];
  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary) return std::nullopt;
  auto lhs = evaluateDimensionExpression(binary.getLHS(), operands, dimensions);
  auto rhs = evaluateDimensionExpression(binary.getRHS(), operands, dimensions);
  if (!lhs || !rhs) return std::nullopt;
  switch (binary.getKind()) {
  case AffineExprKind::Add:
    if (provesSignedNoWrap(BinaryOperator::Add, *lhs, *rhs))
      return intrange::inferAdd({*lhs, *rhs});
    break;
  case AffineExprKind::Mul:
    if (provesSignedNoWrap(BinaryOperator::Multiply, *lhs, *rhs))
      return intrange::inferMul({*lhs, *rhs});
    break;
  case AffineExprKind::FloorDiv:
    if (rhs->smin().isStrictlyPositive())
      return intrange::inferFloorDivS({*lhs, *rhs});
    break;
  case AffineExprKind::CeilDiv:
    if (rhs->smin().isStrictlyPositive())
      return intrange::inferCeilDivS({*lhs, *rhs});
    break;
  case AffineExprKind::Mod:
    if (rhs->smin().isStrictlyPositive()) return floorRemainder(*lhs, *rhs);
    break;
  default:
    break;
  }
  return std::nullopt;
}

} // namespace

IntegerRangeAnalysis::IntegerRangeAnalysis(IntegerRangePolicy policy)
    : policy(std::move(policy)) {}

Type IntegerRangeAnalysis::elementType(Value value) const {
  if (policy.elementType) return policy.elementType(value);
  return getElementTypeOrSelf(value.getType());
}

std::optional<ConstantIntRanges> IntegerRangeAnalysis::range(Value value) {
  if (!value) return std::nullopt;
  unsigned width = bitWidth(elementType(value));
  if (!width) return std::nullopt;
  if (auto found = known.find(value); found != known.end()) return found->second;
  if (!active.insert(value).second) return ConstantIntRanges::maxRange(width);
  auto cleanup = llvm::make_scope_exit([&] { active.erase(value); });
  auto result = infer(value);
  if (!result || result->smin().getBitWidth() != width)
    result = ConstantIntRanges::maxRange(width);
  known.try_emplace(value, *result);
  return result;
}

std::optional<ConstantIntRanges> IntegerRangeAnalysis::infer(Value value) {
  if (policy.infer)
    if (auto result = policy.infer(value, *this)) return result;
  unsigned width = bitWidth(elementType(value));
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return ConstantIntRanges::constant(constant.sextOrTrunc(width));

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    // These SCF counted loops have exclusive upper bounds. LoopLike supplies
    // exact induction-variable slots; it does not itself define bound polarity.
    if (isa<scf::ForOp, scf::ParallelOp, scf::ForallOp>(owner)) {
      auto loop = cast<LoopLikeOpInterface>(owner);
      auto induction = loop.getLoopInductionVars();
      auto lowerBounds = loop.getLoopLowerBounds();
      auto upperBounds = loop.getLoopUpperBounds();
      if (induction && lowerBounds && upperBounds) {
        auto position = llvm::find(*induction, argument);
        if (position != induction->end()) {
          unsigned axis = position - induction->begin();
          if (axis >= lowerBounds->size() || axis >= upperBounds->size())
            return std::nullopt;
          auto bound = [&](OpFoldResult value)
              -> std::optional<ConstantIntRanges> {
            if (auto scalar = dyn_cast<Value>(value)) return range(scalar);
            if (auto integer = dyn_cast<IntegerAttr>(cast<Attribute>(value)))
              return ConstantIntRanges::constant(
                  integer.getValue().sextOrTrunc(width));
            return std::nullopt;
          };
          auto lower = bound((*lowerBounds)[axis]);
          auto upper = bound((*upperBounds)[axis]);
          if (lower && upper && lower->smin().slt(upper->smax())) {
            APInt last = upper->smax() - 1;
            // This is the scoped iteration-domain contract, not a no-wrap
            // assertion about an independently materialized IV + step.
            return ConstantIntRanges::fromSigned(lower->smin(), last);
          }
          return std::nullopt;
        }
      }
    }
  }

  auto incoming = queryControlFlowIncoming(value);
  if (incoming.complete && !incoming.edges.empty()) {
    std::optional<ConstantIntRanges> combined;
    for (const auto &edge : incoming.edges) {
      if (!edge.operand) return std::nullopt;
      auto source = range(edge.operand->get());
      if (!source || source->smin().getBitWidth() != width) return std::nullopt;
      combined = combined ? combined->rangeUnion(*source) : source;
    }
    return combined;
  }

  Operation *operation = value.getDefiningOp();
  if (!operation) return std::nullopt;
  if (auto dim = dyn_cast<memref::DimOp>(operation)) {
    auto axis = dim.getConstantIndex();
    return axis ? dimension(dim.getSource(), *axis) : nonNegativeDimension();
  }
  if (auto dim = dyn_cast<tensor::DimOp>(operation)) {
    auto axis = dim.getConstantIndex();
    return axis ? dimension(dim.getSource(), *axis) : nonNegativeDimension();
  }
  // Native index-cast inference accounts for both 32- and 64-bit targets.
  // Intent index always has 64 bits, so use the bit-width primitive directly.
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(operation)) {
    auto input = range(operation->getOperand(0));
    if (!input) return std::nullopt;
    unsigned from = input->smin().getBitWidth();
    if (from == width) return input;
    if (from > width) return intrange::truncRange(*input, width);
    return isa<arith::IndexCastUIOp>(operation)
               ? intrange::extUIRange(*input, width)
               : intrange::extSIRange(*input, width);
  }
  // Region forwarding is queried above by exact operand slots. Do not invoke
  // an interface that may inspect an as-yet unbuilt terminator in construction.
  auto interface = dyn_cast<InferIntRangeInterface>(operation);
  if (!interface || operation->getNumRegions()) return std::nullopt;
  SmallVector<IntegerValueRange> operands;
  for (Value operand : operation->getOperands())
    operands.emplace_back(range(operand));
  std::optional<ConstantIntRanges> result;
  interface.inferResultRangesFromOptional(
      operands, [&](Value output, const IntegerValueRange &inferred) {
        if (output == value && !inferred.isUninitialized())
          result = inferred.getValue();
      });
  return result;
}

std::optional<ConstantIntRanges>
IntegerRangeAnalysis::dimension(Value shaped, unsigned axis) {
  auto type = dyn_cast<ShapedType>(shaped.getType());
  if (!type || !type.hasRank() || axis >= type.getRank()) return std::nullopt;
  if (!type.isDynamicDim(axis))
    return ConstantIntRanges::constant(APInt(64, type.getDimSize(axis)));
  auto key = std::make_pair(shaped, axis);
  if (!activeDimensions.insert(key).second) return nonNegativeDimension();
  auto cleanup = llvm::make_scope_exit([&] { activeDimensions.erase(key); });
  AffineMap expression;
  ValueDimList leaves;
  if (failed(ValueBoundsConstraintSet::computeBound(
          expression, leaves, presburger::BoundType::EQ,
          ValueBoundsConstraintSet::Variable(shaped, axis), stopAtDimensionLeaf)))
    return nonNegativeDimension();
  SmallVector<ConstantIntRanges> operands;
  for (auto [leaf, dimensionIndex] : leaves) {
    auto bounds = dimensionIndex ? dimension(leaf, *dimensionIndex) : range(leaf);
    if (!bounds || bounds->smin().getBitWidth() != 64)
      return nonNegativeDimension();
    operands.push_back(*bounds);
  }
  auto bounds = evaluateDimensionExpression(expression.getResult(0), operands,
                                            expression.getNumDims());
  if (!bounds || bounds->smax().isNegative()) return nonNegativeDimension();
  return bounds->intersection(nonNegativeDimension());
}

bool IntegerRangeAnalysis::isNonNegative(Value value) {
  auto result = range(value);
  return result && (isUnsigned(elementType(value)) || !result->smin().isNegative());
}

bool IntegerRangeAnalysis::isPositive(Value value) {
  auto result = range(value);
  return result && (isUnsigned(elementType(value))
                        ? !result->umin().isZero()
                        : result->smin().isStrictlyPositive());
}

std::optional<ConstantIntRanges>
inferIntegerBinary(BinaryOperator kind, Type type, const ConstantIntRanges &lhs,
                   const ConstantIntRanges &rhs) {
  unsigned width = bitWidth(type);
  if (!width || lhs.smin().getBitWidth() != width ||
      rhs.smin().getBitWidth() != width) return std::nullopt;
  bool unsignedValue = isUnsigned(type);
  switch (kind) {
  case BinaryOperator::Add:
    return intrange::inferAdd({lhs, rhs});
  case BinaryOperator::Subtract:
    return intrange::inferSub({lhs, rhs});
  case BinaryOperator::Multiply:
    return intrange::inferMul({lhs, rhs});
  case BinaryOperator::FloorDivide:
    return unsignedValue ? intrange::inferDivU({lhs, rhs})
                         : intrange::inferFloorDivS({lhs, rhs});
  case BinaryOperator::Remainder:
    return unsignedValue ? intrange::inferRemU({lhs, rhs})
                         : floorRemainder(lhs, rhs);
  case BinaryOperator::Maximum:
  case BinaryOperator::MaximumNum:
    return unsignedValue ? intrange::inferMaxU({lhs, rhs})
                         : intrange::inferMaxS({lhs, rhs});
  case BinaryOperator::Minimum:
  case BinaryOperator::MinimumNum:
    return unsignedValue ? intrange::inferMinU({lhs, rhs})
                         : intrange::inferMinS({lhs, rhs});
  case BinaryOperator::LogicalAnd:
    return intrange::inferAnd({truthValue(lhs), truthValue(rhs)});
  case BinaryOperator::LogicalOr:
    return intrange::inferOr({truthValue(lhs), truthValue(rhs)});
  case BinaryOperator::BitwiseAnd:
    return intrange::inferAnd({lhs, rhs});
  case BinaryOperator::BitwiseOr:
    return intrange::inferOr({lhs, rhs});
  case BinaryOperator::BitwiseXor:
    return intrange::inferXor({lhs, rhs});
  case BinaryOperator::LeftShift:
    return intrange::inferShl({lhs, rhs});
  case BinaryOperator::RightShift:
    return unsignedValue ? intrange::inferShrU({lhs, rhs})
                         : intrange::inferShrS({lhs, rhs});
  default:
    return std::nullopt;
  }
}

std::optional<ConstantIntRanges>
inferIntegerUnary(UnaryOperator kind, Type type, const ConstantIntRanges &input) {
  unsigned width = bitWidth(type);
  if (!width || input.smin().getBitWidth() != width) return std::nullopt;
  auto zero = ConstantIntRanges::constant(APInt(width, 0));
  switch (kind) {
  case UnaryOperator::Negate:
    return intrange::inferSub({zero, input});
  case UnaryOperator::Not:
    return intrange::inferXor(
        {truthValue(input), ConstantIntRanges::constant(APInt(1, 1))});
  case UnaryOperator::Abs:
    if (isUnsigned(type) || !input.smin().isNegative()) return input;
    if (input.smin().isMinSignedValue()) return ConstantIntRanges::maxRange(width);
    if (input.smax().isNegative()) return intrange::inferSub({zero, input});
    return ConstantIntRanges::fromSigned(
        APInt(width, 0),
        llvm::APIntOps::smax(-input.smin(), input.smax()));
  default:
    return std::nullopt;
  }
}

std::optional<ConstantIntRanges>
inferIntegerCompare(ComparePredicate predicate, Type type,
                    const ConstantIntRanges &lhs, const ConstantIntRanges &rhs) {
  unsigned width = bitWidth(type);
  if (!width || lhs.smin().getBitWidth() != width ||
      rhs.smin().getBitWidth() != width) return std::nullopt;
  using Predicate = intrange::CmpPredicate;
  Predicate comparison;
  bool unsignedValue = isUnsigned(type);
  switch (predicate) {
  case ComparePredicate::Eq:
    comparison = Predicate::eq;
    break;
  case ComparePredicate::Ne:
    comparison = Predicate::ne;
    break;
  case ComparePredicate::Lt:
    comparison = unsignedValue ? Predicate::ult : Predicate::slt;
    break;
  case ComparePredicate::Le:
    comparison = unsignedValue ? Predicate::ule : Predicate::sle;
    break;
  case ComparePredicate::Gt:
    comparison = unsignedValue ? Predicate::ugt : Predicate::sgt;
    break;
  case ComparePredicate::Ge:
    comparison = unsignedValue ? Predicate::uge : Predicate::sge;
    break;
  }
  auto result = intrange::evaluatePred(comparison, lhs, rhs);
  return result ? ConstantIntRanges::constant(APInt(1, *result))
                : ConstantIntRanges::maxRange(1);
}

std::optional<ConstantIntRanges>
inferIntegerCast(Type source, Type target, const ConstantIntRanges &input) {
  unsigned from = bitWidth(source), to = bitWidth(target);
  if (!from || !to || input.smin().getBitWidth() != from) return std::nullopt;
  if (to == 1) return truthValue(input);
  if (from == to) return input;
  if (from > to) return intrange::truncRange(input, to);
  return isUnsigned(source) ? intrange::extUIRange(input, to)
                             : intrange::extSIRange(input, to);
}

bool provesSignedNoWrap(BinaryOperator kind, const ConstantIntRanges &lhs,
                       const ConstantIntRanges &rhs) {
  unsigned width = lhs.smin().getBitWidth();
  if (rhs.smin().getBitWidth() != width) return false;
  unsigned extendedWidth = 2 * width + 1;
  APInt low = lhs.smin().sext(extendedWidth);
  APInt high = lhs.smax().sext(extendedWidth);
  APInt otherLow = rhs.smin().sext(extendedWidth);
  APInt otherHigh = rhs.smax().sext(extendedWidth);
  switch (kind) {
  case BinaryOperator::Add:
    low += otherLow;
    high += otherHigh;
    break;
  case BinaryOperator::Subtract:
    low -= otherHigh;
    high -= otherLow;
    break;
  case BinaryOperator::Multiply: {
    SmallVector<APInt> products{low * otherLow, low * otherHigh,
                               high * otherLow, high * otherHigh};
    auto extrema = std::minmax_element(
        products.begin(), products.end(),
        [](const APInt &a, const APInt &b) { return a.slt(b); });
    low = *extrema.first;
    high = *extrema.second;
    break;
  }
  default:
    return false;
  }
  return low.isSignedIntN(width) && high.isSignedIntN(width);
}

bool isValuePreservingIntegerCast(Type source, Type target,
                                 const ConstantIntRanges &input) {
  unsigned from = bitWidth(source), to = bitWidth(target);
  if (!from || !to || input.smin().getBitWidth() != from) return false;
  unsigned extendedWidth = std::max(from, to) + 1;
  APInt low = isUnsigned(source) ? input.umin().zext(extendedWidth)
                                : input.smin().sext(extendedWidth);
  APInt high = isUnsigned(source) ? input.umax().zext(extendedWidth)
                                 : input.smax().sext(extendedWidth);
  if (isUnsigned(target))
    return !low.isNegative() &&
           high.ule(APInt::getMaxValue(to).zext(extendedWidth));
  return low.isSignedIntN(to) && high.isSignedIntN(to);
}

} // namespace intent
