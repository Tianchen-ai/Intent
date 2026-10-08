#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

using namespace mlir;

namespace intent::gpu {
namespace {

std::optional<bool> booleanJunction(BinaryOp operation) {
  Type element = operation.getResult().getType();
  if (auto fragment = dyn_cast<FragmentType>(element))
    element = fragment.getElementType();
  if (!element.isInteger(1) ||
      operation.getLhs().getType() != operation.getResult().getType() ||
      operation.getRhs().getType() != operation.getResult().getType())
    return std::nullopt;
  switch (operation.getOperatorKind()) {
  case BinaryOperator::LogicalAnd:
  case BinaryOperator::BitwiseAnd: return true;
  case BinaryOperator::LogicalOr:
  case BinaryOperator::BitwiseOr: return false;
  default: return std::nullopt;
  }
}

std::optional<bool> projectedBooleanConstant(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    if (auto constant = dyn_cast<arith::ConstantOp>(definition)) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      if (integer && integer.getType().isInteger(1))
        return integer.getValue().isOne();
      return std::nullopt;
    }
    // These pure projections preserve a uniform value in every physical lane.
    // A numerical cast or a nonuniform producer is not a constant proof.
    if (!isa<SplatOp, BroadcastOp, ReshapeOp, TransposeOp>(definition))
      return std::nullopt;
    value = definition->getOperand(0);
  }
  return std::nullopt;
}

} // namespace

Operation *IntentGPUDialect::materializeConstant(OpBuilder &builder,
    Attribute value, Type type, Location location) {
  if (!arith::ConstantOp::isBuildableWith(value, type)) return nullptr;
  return builder.create<arith::ConstantOp>(location, type, cast<TypedAttr>(value));
}

OpFoldResult BroadcastOp::fold(FoldAdaptor) {
  return getValue().getType() == getResult().getType()
      ? OpFoldResult(getValue()) : OpFoldResult();
}
OpFoldResult CastOp::fold(FoldAdaptor) {
  return getValue().getType() == getResult().getType()
      ? OpFoldResult(getValue()) : OpFoldResult();
}
OpFoldResult BitcastOp::fold(FoldAdaptor) {
  return getValue().getType() == getResult().getType()
      ? OpFoldResult(getValue()) : OpFoldResult();
}
OpFoldResult ReshapeOp::fold(FoldAdaptor) {
  return getValue().getType() == getResult().getType()
      ? OpFoldResult(getValue()) : OpFoldResult();
}
OpFoldResult TransposeOp::fold(FoldAdaptor) {
  if (getValue().getType() != getResult().getType()) return {};
  for (auto [axis, source] : llvm::enumerate(getPermutation()))
    if (axis != static_cast<unsigned>(source)) return {};
  return getValue();
}
OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  Value selected;
  if (getTrueValue() == getFalseValue()) selected = getTrueValue();
  else if (auto condition = dyn_cast_or_null<IntegerAttr>(adaptor.getCondition());
           condition && condition.getType().isInteger(1))
    selected = condition.getValue().isZero() ? getFalseValue() : getTrueValue();
  if (selected && selected.getType() == getResult().getType()) return selected;
  return {};
}
OpFoldResult BinaryOp::fold(FoldAdaptor adaptor) {
  Type result = getResult().getType();
  if (auto conjunction = booleanJunction(*this)) {
    if (getLhs() == getRhs()) return getLhs();
    for (unsigned side = 0; side < 2; ++side)
      if (auto constant = projectedBooleanConstant(getOperand(side)))
        return *constant == *conjunction ? getOperand(1 - side)
                                        : getOperand(side);
  }
  if (result.isIntOrIndex()) {
    auto lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
    auto rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
    if (!lhs || !rhs || lhs.getType() != result || rhs.getType() != result)
      return {};
    APInt left = lhs.getValue(), right = rhs.getValue(), value = left;
    auto integer = dyn_cast<IntegerType>(result);
    bool unsignedType = integer &&
        (integer.isUnsigned() || integer.getWidth() == 1);
    switch (getOperatorKind()) {
    case BinaryOperator::Add: value += right; break;
    case BinaryOperator::Subtract: value -= right; break;
    case BinaryOperator::Multiply: value *= right; break;
    case BinaryOperator::FloorDivide:
    case BinaryOperator::Remainder: {
      if (right.isZero() || (!unsignedType && left.isMinSignedValue() &&
                             right.isAllOnes())) return {};
      APInt quotient = unsignedType ? left.udiv(right) : left.sdiv(right);
      APInt remainder = unsignedType ? left.urem(right) : left.srem(right);
      if (!unsignedType && !remainder.isZero() &&
          left.isNegative() != right.isNegative()) {
        --quotient;
        remainder += right;
      }
      value = getOperatorKind() == BinaryOperator::FloorDivide
                  ? quotient : remainder;
      break;
    }
    case BinaryOperator::LogicalAnd:
      value = APInt(left.getBitWidth(), !left.isZero() && !right.isZero());
      break;
    case BinaryOperator::Maximum:
    case BinaryOperator::MaximumNum:
      value = (unsignedType ? left.ult(right) : left.slt(right)) ? right : left;
      break;
    case BinaryOperator::Minimum:
    case BinaryOperator::MinimumNum:
      value = (unsignedType ? left.ult(right) : left.slt(right)) ? left : right;
      break;
    default: return {};
    }
    return IntegerAttr::get(result, value);
  }
  if (getOperatorKind() != BinaryOperator::TrueDivide || getApproximate() ||
      getFlushToZero()) return {};
  auto type = dyn_cast<FloatType>(getResult().getType());
  auto lhs = dyn_cast_or_null<FloatAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<FloatAttr>(adaptor.getRhs());
  if (!type || !lhs || !rhs || lhs.getType() != type || rhs.getType() != type)
    return {};
  auto quotient = lhs.getValue();
  quotient.divide(rhs.getValue(), llvm::APFloat::rmNearestTiesToEven);
  return FloatAttr::get(type, quotient);
}
OpFoldResult ExtractOp::fold(FoldAdaptor) {
  auto record = getRecord().getDefiningOp<MakeRecordOp>();
  if (!record || getField() >= record.getFields().size()) return {};
  Value field = record.getFields()[getField()];
  return field.getType() == getResult().getType() ? OpFoldResult(field)
                                                : OpFoldResult();
}
OpFoldResult MakeRecordOp::fold(FoldAdaptor) {
  Value source;
  for (auto [index, field] : llvm::enumerate(getFields())) {
    auto extract = field.getDefiningOp<ExtractOp>();
    if (!extract || extract.getField() != index ||
        (source && source != extract.getRecord())) return {};
    source = extract.getRecord();
  }
  return source && source.getType() == getResult().getType()
      ? OpFoldResult(source) : OpFoldResult();
}

namespace {
struct DeduplicateBooleanTerms : OpRewritePattern<BinaryOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BinaryOp operation,
                                PatternRewriter &rewriter) const override {
    auto conjunction = booleanJunction(operation);
    if (!conjunction) return failure();
    SmallVector<Value> pending{operation.getRhs(), operation.getLhs()};
    SmallVector<Value> terms;
    llvm::SmallPtrSet<void *, 16> seen;
    bool duplicate = false;
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      auto nested = value.getDefiningOp<BinaryOp>();
      // Expanding only uniquely used trees avoids copying a shared predicate
      // DAG. The original leaf order remains stable after duplicate removal.
      if (nested && value.hasOneUse() && value.getType() == operation.getType() &&
          booleanJunction(nested) == conjunction) {
        pending.push_back(nested.getRhs());
        pending.push_back(nested.getLhs());
      } else if (seen.insert(value.getAsOpaquePointer()).second) {
        terms.push_back(value);
      } else {
        duplicate = true;
      }
    }
    if (!duplicate) return failure();
    Value replacement = terms.front();
    for (Value term : llvm::drop_begin(terms)) {
      auto combined = rewriter.create<BinaryOp>(operation.getLoc(), operation.getType(),
          replacement, term, operation.getOperatorKind(), operation.getApproximate(),
          operation.getFlushToZero(), operation.getStrictRounding());
      combined->setDiscardableAttrs(llvm::to_vector(operation->getDiscardableAttrs()));
      replacement = combined.getResult();
    }
    rewriter.replaceOp(operation, replacement);
    return success();
  }
};

struct SingletonInsertion : OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    auto relations = queryFragmentOperandRelations(reshape.getOperation());
    if (failed(relations) || !relations->front().isUnitAxisInsertion())
      return failure();
    auto replacement = rewriter.create<BroadcastOp>(
        reshape.getLoc(), reshape.getResult().getType(), reshape.getValue());
    replacement->setDiscardableAttrs(llvm::to_vector(reshape->getDiscardableAttrs()));
    rewriter.replaceOp(reshape, replacement.getResult());
    return success();
  }
};

struct ComposeTranspose : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp outer,
                                PatternRewriter &rewriter) const override {
    auto inner = outer.getValue().getDefiningOp<TransposeOp>();
    if (!inner) return failure();
    SmallVector<int64_t> permutation;
    for (int64_t axis : outer.getPermutation())
      permutation.push_back(inner.getPermutation()[axis]);
    rewriter.modifyOpInPlace(outer, [&] {
      outer.getValueMutable().assign(inner.getValue());
      outer.setPermutationAttr(rewriter.getDenseI64ArrayAttr(permutation));
    });
    return success();
  }
};

struct CombineSelections : OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SelectOp select,
                                PatternRewriter &rewriter) const override {
    for (unsigned arm = 0; arm < 2; ++arm) {
      auto inner = select->getOperand(arm + 1).getDefiningOp<SelectOp>();
      Value otherwise = select->getOperand(2 - arm);
      if (!inner || !inner->hasOneUse()) continue;
      bool inverted = inner.getTrueValue() == otherwise;
      if (!inverted && inner.getFalseValue() != otherwise) continue;
      Location location = select.getLoc();
      Type predicateType = select.getCondition().getType();
      if (inner.getCondition().getType() != predicateType) continue;
      Value outerCondition = select.getCondition();
      Value innerCondition = inner.getCondition();
      if (arm) outerCondition = rewriter.create<UnaryOp>(
          location, predicateType, outerCondition, UnaryOperator::Not);
      if (inverted) innerCondition = rewriter.create<UnaryOp>(
          location, predicateType, innerCondition, UnaryOperator::Not);
      Value inactive = rewriter.create<arith::ConstantOp>(
          location, rewriter.getBoolAttr(false));
      if (auto fragment = dyn_cast<FragmentType>(predicateType))
        inactive = rewriter.create<SplatOp>(location, fragment, inactive);
      Value active = rewriter.create<SelectOp>(
          location, predicateType, outerCondition, innerCondition, inactive);
      rewriter.modifyOpInPlace(select, [&] {
        select->setOperands({active, inverted ? inner.getFalseValue()
                                             : inner.getTrueValue(), otherwise});
      });
      return success();
    }
    return failure();
  }
};
} // namespace

void BinaryOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<DeduplicateBooleanTerms>(context);
}
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add<SingletonInsertion>(context);
}
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<ComposeTranspose>(context);
}
void SelectOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<CombineSelections>(context);
}
} // namespace intent::gpu
