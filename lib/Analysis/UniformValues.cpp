#include "Intent/Analysis/UniformValues.h"
#include "mlir/IR/TypeUtilities.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/APSInt.h"

using namespace mlir;
namespace intent {
namespace {

unsigned bitWidth(Type type) {
  return type.isIndex() ? 64 : cast<IntegerType>(type).getWidth();
}
bool isUnsigned(Type type) {
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.isUnsigned() || integer.getWidth() == 1);
}
Attribute castConstant(Attribute value, Type type, bool unsignedInput, bool unsignedOutput = false) {
  if (auto integer = dyn_cast_or_null<IntegerAttr>(value)) {
    bool unsignedSource = unsignedInput || isUnsigned(integer.getType());
    if (type.isIntOrIndex()) {
      auto bits = unsignedSource ? integer.getValue().zextOrTrunc(bitWidth(type))
                                 : integer.getValue().sextOrTrunc(bitWidth(type));
      return IntegerAttr::get(type, bits);
    }
    if (auto floating = dyn_cast<FloatType>(type)) {
      llvm::APFloat number(floating.getFloatSemantics());
      number.convertFromAPInt(integer.getValue(), !unsignedSource, llvm::APFloat::rmNearestTiesToEven);
      return FloatAttr::get(type, number);
    }
  }
  if (auto floating = dyn_cast_or_null<FloatAttr>(value)) {
    auto number = floating.getValue();
    if (auto target = dyn_cast<FloatType>(type)) {
      bool loses;
      auto status = number.convert(target.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &loses);
      if (status & llvm::APFloat::opInvalidOp) return {};
      return FloatAttr::get(type, number);
    }
    if (type.isIntOrIndex()) {
      llvm::APSInt bits(bitWidth(type), unsignedOutput || isUnsigned(type));
      bool exact;
      auto status = number.convertToInteger(bits, llvm::APFloat::rmTowardZero, &exact);
      if (status & (llvm::APFloat::opInvalidOp | llvm::APFloat::opOverflow)) return {};
      return IntegerAttr::get(type, bits);
    }
  }
  return {};
}
bool numericZero(Attribute value) {
  if (auto integer = dyn_cast_or_null<IntegerAttr>(value)) return integer.getValue().isZero();
  if (auto floating = dyn_cast_or_null<FloatAttr>(value)) return floating.getValue().isZero();
  return false;
}

Attribute arithmetic(const UniformExpression &expression, ArrayRef<Attribute> values) {
  using K = UniformKind;
  Type type = expression.type;
  if ((expression.kind == K::Multiply || expression.kind == K::And) && type.isIntOrIndex() &&
      llvm::any_of(values, numericZero)) return uniformZero(type);
  if (expression.kind == K::Or && type.isInteger(1) &&
      llvm::any_of(values, [](Attribute value) { return uniformBoolean(value) == true; }))
    return IntegerAttr::get(type, 1);
  if (llvm::any_of(values, [](Attribute value) { return !value; })) return {};
  if (expression.kind == K::Cast)
    return castConstant(values[0], type, expression.unsignedInput, expression.unsignedOutput);
  if (expression.kind == K::Bitcast) {
    llvm::APInt bits;
    if (auto integer = dyn_cast<IntegerAttr>(values[0])) bits = integer.getValue();
    else if (auto floating = dyn_cast<FloatAttr>(values[0])) bits = floating.getValue().bitcastToAPInt();
    else return {};
    if (type.isIntOrIndex() && bitWidth(type) == bits.getBitWidth()) return IntegerAttr::get(type, bits);
    if (auto floating = dyn_cast<FloatType>(type); floating && floating.getWidth() == bits.getBitWidth())
      return FloatAttr::get(type, llvm::APFloat(floating.getFloatSemantics(), bits));
    return {};
  }
  if (expression.kind == K::Compare) {
    bool unordered = false;
    int order;
    if (auto left = dyn_cast<IntegerAttr>(values[0])) {
      auto right = dyn_cast<IntegerAttr>(values[1]);
      if (!right || left.getType() != right.getType()) return {};
      bool less = expression.unsignedInput || isUnsigned(left.getType())
          ? left.getValue().ult(right.getValue()) : left.getValue().slt(right.getValue());
      order = left.getValue() == right.getValue() ? 0 : less ? -1 : 1;
    } else if (auto left = dyn_cast<FloatAttr>(values[0])) {
      auto right = dyn_cast<FloatAttr>(values[1]);
      if (!right || left.getType() != right.getType()) return {};
      auto compared = left.getValue().compare(right.getValue());
      unordered = compared == llvm::APFloat::cmpUnordered;
      order = compared == llvm::APFloat::cmpEqual ? 0 : compared == llvm::APFloat::cmpLessThan ? -1 : 1;
    } else return {};
    bool result = false;
    switch (expression.predicate) {
    case UniformPredicate::Ordered: result = !unordered; break;
    case UniformPredicate::Unordered: result = unordered; break;
    case UniformPredicate::Equal: result = unordered ? expression.unorderedTrue : order == 0; break;
    case UniformPredicate::NotEqual: result = unordered ? expression.unorderedTrue : order != 0; break;
    case UniformPredicate::Less: result = unordered ? expression.unorderedTrue : order < 0; break;
    case UniformPredicate::LessEqual: result = unordered ? expression.unorderedTrue : order <= 0; break;
    case UniformPredicate::Greater: result = unordered ? expression.unorderedTrue : order > 0; break;
    case UniformPredicate::GreaterEqual: result = unordered ? expression.unorderedTrue : order >= 0; break;
    }
    return IntegerAttr::get(type, result);
  }
  if (type.isIntOrIndex()) {
    auto left = dyn_cast<IntegerAttr>(values[0]);
    if (!left || left.getType() != type) return {};
    llvm::APInt result = left.getValue();
    if (expression.kind == K::Negate) return IntegerAttr::get(type, -result);
    if (expression.kind == K::Not) return IntegerAttr::get(type, ~result);
    if (values.size() != 2) return {};
    auto right = dyn_cast<IntegerAttr>(values[1]);
    if (!right || right.getType() != type) return {};
    switch (expression.kind) {
    case K::Add: result += right.getValue(); break;
    case K::Subtract: result -= right.getValue(); break;
    case K::Multiply: result *= right.getValue(); break;
    case K::And: result &= right.getValue(); break;
    case K::Or: result |= right.getValue(); break;
    case K::Xor: result ^= right.getValue(); break;
    case K::Maximum: case K::MaximumNum: case K::Minimum: case K::MinimumNum: {
      bool less = isUnsigned(type) ? result.ult(right.getValue()) : result.slt(right.getValue());
      bool maximum = expression.kind == K::Maximum || expression.kind == K::MaximumNum;
      return less == maximum ? Attribute(right) : Attribute(left);
    }
    default: return {};
    }
    return IntegerAttr::get(type, result);
  }
  auto left = dyn_cast<FloatAttr>(values[0]);
  if (!left || left.getType() != type) return {};
  auto result = left.getValue();
  if (expression.kind == K::Negate) { result.changeSign(); return FloatAttr::get(type, result); }
  if (expression.kind == K::Exp) {
    if (result.isZero()) return FloatAttr::get(type, llvm::APFloat::getOne(result.getSemantics()));
    if (result.isNaN() || (result.isInfinity() && !result.isNegative())) return left;
    return result.isInfinity() ? uniformZero(type) : Attribute();
  }
  if (values.size() != 2) return {};
  auto right = dyn_cast<FloatAttr>(values[1]);
  if (!right || right.getType() != type) return {};
  switch (expression.kind) {
  case K::Add: result.add(right.getValue(), llvm::APFloat::rmNearestTiesToEven); break;
  case K::Subtract: result.subtract(right.getValue(), llvm::APFloat::rmNearestTiesToEven); break;
  case K::Multiply: result.multiply(right.getValue(), llvm::APFloat::rmNearestTiesToEven); break;
  case K::Maximum: case K::Minimum: case K::MaximumNum: case K::MinimumNum: {
    bool maximum = expression.kind == K::Maximum || expression.kind == K::MaximumNum;
    bool number = expression.kind == K::MaximumNum || expression.kind == K::MinimumNum;
    if (result.isNaN()) return number ? Attribute(right) : Attribute(left);
    if (right.getValue().isNaN()) return number ? Attribute(left) : Attribute(right);
    if (result.isZero() && right.getValue().isZero()) {
      bool negative = maximum ? result.isNegative() && right.getValue().isNegative()
                              : result.isNegative() || right.getValue().isNegative();
      return FloatAttr::get(type, llvm::APFloat::getZero(result.getSemantics(), negative));
    }
    auto comparison = result.compare(right.getValue());
    return (maximum ? comparison == llvm::APFloat::cmpLessThan : comparison == llvm::APFloat::cmpGreaterThan)
        ? Attribute(right) : Attribute(left);
  }
  default: return {};
  }
  return FloatAttr::get(type, result);
}

}

bool equalUniformConstants(Attribute lhs, Attribute rhs) {
  if (!lhs || !rhs) return false;
  if (auto left = dyn_cast<FloatAttr>(lhs)) {
    auto right = dyn_cast<FloatAttr>(rhs);
    return right && left.getType() == right.getType() &&
        ((left.getValue().isNaN() && right.getValue().isNaN()) || left.getValue().bitwiseIsEqual(right.getValue()));
  }
  if (auto left = dyn_cast<ArrayAttr>(lhs)) {
    auto right = dyn_cast<ArrayAttr>(rhs);
    return right && left.size() == right.size() && llvm::all_of(llvm::zip(left, right), [](auto pair) {
      return equalUniformConstants(std::get<0>(pair), std::get<1>(pair));
    });
  }
  return lhs == rhs;
}
Attribute uniformZero(Type type) {
  if (type.isIntOrIndex()) return IntegerAttr::get(type, 0);
  if (auto floating = dyn_cast<FloatType>(type))
    return FloatAttr::get(type, llvm::APFloat::getZero(floating.getFloatSemantics()));
  return {};
}
std::optional<bool> uniformBoolean(Attribute value) {
  auto integer = dyn_cast_or_null<IntegerAttr>(value);
  if (!integer || !integer.getType().isInteger(1)) return std::nullopt;
  return !integer.getValue().isZero();
}

Attribute UniformValueAnalysis::evaluate(Value value, const UniformBindings &bindings) const {
  llvm::SmallDenseSet<Value> visiting;
  return evaluate(value, bindings, visiting);
}
Attribute UniformValueAnalysis::fold(const UniformExpression &expression, const UniformBindings &bindings) const {
  llvm::SmallDenseSet<Value> visiting;
  return fold(expression, bindings, visiting);
}
Attribute UniformValueAnalysis::evaluate(Value value, const UniformBindings &bindings,
                                        llvm::SmallDenseSet<Value> &visiting) const {
  auto found = bindings.find(value);
  if (found != bindings.end()) return found->second;
  if (!visiting.insert(value).second) return {};
  Attribute result = fold(describe(value), bindings, visiting);
  visiting.erase(value);
  return result;
}
Attribute UniformValueAnalysis::fold(const UniformExpression &expression, const UniformBindings &bindings,
                                    llvm::SmallDenseSet<Value> &visiting) const {
  using K = UniformKind;
  if (expression.kind == K::Unknown) return {};
  if (expression.kind == K::Constant) return expression.literal;
  auto get = [&](Value value) { return evaluate(value, bindings, visiting); };
  if (expression.kind == K::Forward) return get(expression.operands[0]);
  if (expression.kind == K::Select) {
    auto condition = uniformBoolean(get(expression.operands[0]));
    if (condition) return get(expression.operands[*condition ? 1 : 2]);
    Attribute lhs = get(expression.operands[1]), rhs = get(expression.operands[2]);
    return equalUniformConstants(lhs, rhs) ? lhs : Attribute();
  }
  if (expression.kind == K::Extract) {
    auto record = dyn_cast_or_null<ArrayAttr>(get(expression.operands[0]));
    return record && expression.result < record.size() ? record[expression.result] : Attribute();
  }
  if (expression.kind == K::Fold) {
    SmallVector<Attribute> initial, inputs;
    for (Value value : ArrayRef(expression.operands).take_front(expression.stateCount)) {
      Attribute constant = get(value);
      if (!constant) return {};
      initial.push_back(constant);
    }
    for (Value value : ArrayRef(expression.operands).drop_front(expression.stateCount)) inputs.push_back(get(value));
    auto step = [&](ArrayRef<Attribute> state) {
      UniformBindings current = bindings;
      for (auto [parameter, value] : llvm::zip(ArrayRef(expression.parameters).take_front(state.size()), state))
        current[parameter] = value;
      for (auto [parameter, value] : llvm::zip(ArrayRef(expression.parameters).drop_front(state.size()), inputs))
        current[parameter] = value;
      SmallVector<Attribute> result;
      for (Value value : expression.yields) result.push_back(evaluate(value, current, visiting));
      return result;
    };
    auto first = step(initial);
    if (first.size() != initial.size() || llvm::any_of(first, [](Attribute value) { return !value; })) return {};
    auto equal = [](ArrayRef<Attribute> a, ArrayRef<Attribute> b) {
      return a.size() == b.size() && llvm::all_of(llvm::zip(a, b), [](auto pair) {
        return equalUniformConstants(std::get<0>(pair), std::get<1>(pair));
      });
    };
    if (!expression.nonempty && !equal(first, initial)) return {};
    if (!equal(first, step(first))) return {};
    return expression.result < first.size() ? first[expression.result] : Attribute();
  }
  SmallVector<Attribute> values;
  for (Value value : expression.operands) values.push_back(get(value));
  if (expression.kind == K::Aggregate) {
    if (llvm::any_of(values, [](Attribute value) { return !value; })) return {};
    return ArrayAttr::get(expression.type.getContext(), values);
  }
  if (expression.kind == K::Join)
    return equalUniformConstants(values[0], values[1]) ? values[0] : Attribute();
  if (expression.kind == K::Contract) {
    Attribute initial = values[2];
    if (!initial) return {};
    Attribute lhs = castConstant(values[0], expression.type, false);
    Attribute rhs = castConstant(values[1], expression.type, false);
    if (expression.type.isIntOrIndex()) {
      if (numericZero(lhs) || numericZero(rhs)) return initial;
      return {};
    }
    auto a = dyn_cast_or_null<FloatAttr>(lhs), b = dyn_cast_or_null<FloatAttr>(rhs);
    auto accumulator = dyn_cast<FloatAttr>(initial);
    if (!a || !b || !accumulator || !numericZero(initial) || (!numericZero(lhs) && !numericZero(rhs))) return {};
    auto next = a.getValue();
    next.fusedMultiplyAdd(b.getValue(), accumulator.getValue(), llvm::APFloat::rmNearestTiesToEven);
    Attribute result = FloatAttr::get(expression.type, next);
    return equalUniformConstants(initial, result) ? initial : Attribute();
  }
  return arithmetic(expression, values);
}

UniformExpression describeUniformUnary(Type type, UnaryOperator kind,
    Value input, bool approximate, bool flushToZero) {
  UniformExpression result;
  result.type = type;
  result.operands = {input};
  if (approximate || flushToZero) return result;
  using K = UniformKind;
  switch (kind) {
  case UnaryOperator::Negate: result.kind = K::Negate; break;
  case UnaryOperator::Not: result.kind = K::Not; break;
  case UnaryOperator::Exp: case UnaryOperator::Exp2: result.kind = K::Exp; break;
  default: break;
  }
  return result;
}

UniformExpression describeUniformBinary(Type type, BinaryOperator kind,
    Value lhs, Value rhs, bool approximate, bool flushToZero) {
  UniformExpression result;
  result.type = type;
  result.operands = {lhs, rhs};
  if (approximate || flushToZero) return result;
  using K = UniformKind;
  switch (kind) {
  case BinaryOperator::Add: result.kind = K::Add; break;
  case BinaryOperator::Subtract: result.kind = K::Subtract; break;
  case BinaryOperator::Multiply: result.kind = K::Multiply; break;
  case BinaryOperator::LogicalAnd: case BinaryOperator::BitwiseAnd: result.kind = K::And; break;
  case BinaryOperator::LogicalOr: case BinaryOperator::BitwiseOr: result.kind = K::Or; break;
  case BinaryOperator::BitwiseXor: result.kind = K::Xor; break;
  case BinaryOperator::Maximum: result.kind = K::Maximum; break;
  case BinaryOperator::Minimum: result.kind = K::Minimum; break;
  case BinaryOperator::MaximumNum: result.kind = K::MaximumNum; break;
  case BinaryOperator::MinimumNum: result.kind = K::MinimumNum; break;
  default: break;
  }
  return result;
}

UniformExpression describeUniformCompare(Type type, Type operandType,
    ComparePredicate predicate, Value lhs, Value rhs) {
  UniformExpression result;
  result.type = type;
  result.operands = {lhs, rhs};
  result.kind = UniformKind::Compare;
  result.unsignedInput = isUnsigned(operandType);
  switch (predicate) {
  case ComparePredicate::Eq: result.predicate = UniformPredicate::Equal; break;
  case ComparePredicate::Ne:
    result.predicate = UniformPredicate::NotEqual;
    result.unorderedTrue = true;
    break;
  case ComparePredicate::Lt: result.predicate = UniformPredicate::Less; break;
  case ComparePredicate::Le: result.predicate = UniformPredicate::LessEqual; break;
  case ComparePredicate::Gt: result.predicate = UniformPredicate::Greater; break;
  case ComparePredicate::Ge: result.predicate = UniformPredicate::GreaterEqual; break;
  }
  return result;
}

UniformExpression describeCanonicalUniformValue(Value value) {
  UniformExpression result = describeScalarValue(value);
  if (auto tensor = dyn_cast<RankedTensorType>(value.getType()))
    result.type = tensor.getElementType();
  Operation *op = value.getDefiningOp();
  if (!op) return result;
  using K = UniformKind;
  result.operands.assign(op->operand_begin(), op->operand_end());
  if (auto constant = dyn_cast<ConstantOp>(op)) {
    result.kind = K::Constant;
    result.literal = castConstant(constant.getValue(), result.type, false);
  } else if (isa<BroadcastOp, FullOp, ReshapeOp, TransposeOp>(op)) result.kind = K::Forward;
  else if (auto gather = dyn_cast<GatherOp>(op)) {
    auto valid = gather.getValid();
    auto fill = gather.getFill();
    if (valid && fill) {
      result.kind = K::Select;
      result.operands = {valid, gather.getSource(), fill};
    } else if (!valid) {
      result.kind = K::Forward;
      result.operands = {gather.getSource()};
    }
  }
  else if (isa<MakeRecordOp, MakeTupleOp>(op)) result.kind = K::Aggregate;
  else if (auto extract = dyn_cast<ExtractOp>(op)) {
    result.kind = K::Extract; result.result = extract.getField();
  } else if (isa<SelectOp>(op)) result.kind = K::Select;
  else if (auto mask = dyn_cast<MaskOp>(op)) {
    result.kind = K::Select;
    result.operands = {mask.getPredicate(), mask.getValue(), mask.getFill()};
  }
  else if (isa<CastOp>(op)) result.kind = K::Cast;
  else if (isa<BitcastOp>(op)) result.kind = K::Bitcast;
  else if (auto unary = dyn_cast<UnaryOp>(op)) {
    return describeUniformUnary(result.type, unary.getOperatorKind(), unary.getInput(),
                                 unary.getApproximate(), unary.getFlushToZero());
  } else if (auto binary = dyn_cast<BinaryOp>(op)) {
    return describeUniformBinary(result.type, binary.getOperatorKind(), binary.getLhs(), binary.getRhs(),
                                  binary.getApproximate(), binary.getFlushToZero());
  } else if (auto compare = dyn_cast<CompareOp>(op)) {
    return describeUniformCompare(result.type, getElementTypeOrSelf(compare.getLhs().getType()),
                                   compare.getPredicate(), compare.getLhs(), compare.getRhs());
  } else if (isa<ReduceOp>(op))
    return describeStructuredReduction(cast<OpResult>(value), result.type);
  return result;
}

UniformExpression describeStructuredReduction(OpResult value, Type elementType,
                                              bool nonempty) {
  UniformExpression result;
  result.type = elementType;
  auto operation = dyn_cast<StructuredOpInterface>(value.getOwner());
  if (!operation || operation.getStructuredKind() != StructuredOpKind::Reduce ||
      operation.getSources().size() != operation.getIdentities().size())
    return result;
  result.kind = UniformKind::Fold;
  result.stateCount = operation.getIdentities().size();
  result.result = value.getResultNumber();
  result.nonempty = nonempty;
  llvm::append_range(result.operands, operation.getIdentities());
  llvm::append_range(result.operands, operation.getSources());
  llvm::append_range(result.operands, operation.getCaptures());
  llvm::append_range(result.parameters, operation.getCombineLhs());
  llvm::append_range(result.parameters, operation.getCombineRhs());
  llvm::append_range(result.parameters, operation.getCombineCaptures());
  llvm::append_range(result.yields, operation.getCombineYields());
  return result;
}

UniformExpression describeScalarValue(Value value) {
  UniformExpression result;
  result.type = value.getType();
  Operation *op = value.getDefiningOp();
  if (!op) return result;
  using K = UniformKind;
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    result.kind = K::Constant; result.literal = constant.getValue(); return result;
  }
  result.operands.assign(op->operand_begin(), op->operand_end());
  if (isa<arith::AddIOp, arith::AddFOp>(op)) result.kind = K::Add;
  else if (isa<arith::SubIOp, arith::SubFOp>(op)) result.kind = K::Subtract;
  else if (isa<arith::MulIOp, arith::MulFOp>(op)) result.kind = K::Multiply;
  else if (isa<arith::NegFOp>(op)) result.kind = K::Negate;
  else if (isa<arith::AndIOp>(op)) result.kind = K::And;
  else if (isa<arith::OrIOp>(op)) result.kind = K::Or;
  else if (isa<arith::XOrIOp>(op)) result.kind = K::Xor;
  else if (isa<arith::SelectOp>(op)) result.kind = K::Select;
  else if (isa<arith::MaximumFOp>(op)) result.kind = K::Maximum;
  else if (isa<arith::MinimumFOp>(op)) result.kind = K::Minimum;
  else if (isa<arith::MaxNumFOp>(op)) result.kind = K::MaximumNum;
  else if (isa<arith::MinNumFOp>(op)) result.kind = K::MinimumNum;
  else if (isa<math::ExpOp, math::Exp2Op>(op)) result.kind = K::Exp;
  else if (isa<arith::BitcastOp>(op)) result.kind = K::Bitcast;
  else if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp,
               arith::TruncIOp, arith::ExtFOp, arith::TruncFOp, arith::SIToFPOp,
               arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp>(op)) {
    result.kind = K::Cast;
    result.unsignedInput = isa<arith::IndexCastUIOp, arith::ExtUIOp, arith::UIToFPOp>(op);
    result.unsignedOutput = isa<arith::FPToUIOp>(op);
  } else if (auto compare = dyn_cast<arith::CmpIOp>(op)) {
    result.kind = K::Compare;
    switch (compare.getPredicate()) {
    case arith::CmpIPredicate::eq: result.predicate = UniformPredicate::Equal; break;
    case arith::CmpIPredicate::ne: result.predicate = UniformPredicate::NotEqual; break;
    case arith::CmpIPredicate::slt: case arith::CmpIPredicate::ult: result.predicate = UniformPredicate::Less; break;
    case arith::CmpIPredicate::sle: case arith::CmpIPredicate::ule: result.predicate = UniformPredicate::LessEqual; break;
    case arith::CmpIPredicate::sgt: case arith::CmpIPredicate::ugt: result.predicate = UniformPredicate::Greater; break;
    case arith::CmpIPredicate::sge: case arith::CmpIPredicate::uge: result.predicate = UniformPredicate::GreaterEqual; break;
    }
    result.unsignedInput = compare.getPredicate() >= arith::CmpIPredicate::ult;
  } else if (auto compare = dyn_cast<arith::CmpFOp>(op)) {
    result.kind = K::Compare;
    switch (compare.getPredicate()) {
    case arith::CmpFPredicate::OEQ: case arith::CmpFPredicate::UEQ: result.predicate = UniformPredicate::Equal; break;
    case arith::CmpFPredicate::ONE: case arith::CmpFPredicate::UNE: result.predicate = UniformPredicate::NotEqual; break;
    case arith::CmpFPredicate::OLT: case arith::CmpFPredicate::ULT: result.predicate = UniformPredicate::Less; break;
    case arith::CmpFPredicate::OLE: case arith::CmpFPredicate::ULE: result.predicate = UniformPredicate::LessEqual; break;
    case arith::CmpFPredicate::OGT: case arith::CmpFPredicate::UGT: result.predicate = UniformPredicate::Greater; break;
    case arith::CmpFPredicate::OGE: case arith::CmpFPredicate::UGE: result.predicate = UniformPredicate::GreaterEqual; break;
    case arith::CmpFPredicate::ORD: result.predicate = UniformPredicate::Ordered; break;
    case arith::CmpFPredicate::UNO: result.predicate = UniformPredicate::Unordered; break;
    case arith::CmpFPredicate::AlwaysFalse: case arith::CmpFPredicate::AlwaysTrue:
      result.kind = K::Constant;
      result.literal = IntegerAttr::get(result.type, compare.getPredicate() == arith::CmpFPredicate::AlwaysTrue);
      break;
    }
    result.unorderedTrue = compare.getPredicate() >= arith::CmpFPredicate::UEQ;
  }
  return result;
}

}
