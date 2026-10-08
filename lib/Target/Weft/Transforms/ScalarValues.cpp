#include "ScalarValues.h"
#include "Intent/Serialization/Scalar.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
namespace {

Type elementType(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return value.getElementType();
  return type;
}

Type withElementType(Type type, Type element) {
  if (auto value = dyn_cast<wk::ValueType>(type))
    return wk::ValueType::get(type.getContext(), element, value.getShape(),
                              value.getAxisIds());
  return element;
}

unsigned integerWidth(Type type) {
  type = elementType(type);
  return type.isIndex() ? 64 : cast<IntegerType>(type).getWidth();
}

} // namespace

Type nativeScalarType(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.isSignless() && integer.getWidth() != 1)
    return IntegerType::get(type.getContext(), integer.getWidth(),
                            IntegerType::Signed);
  return type;
}

Type nativeValueType(Type element, ArrayRef<int64_t> shape,
                     ArrayRef<int64_t> axes) {
  element = nativeScalarType(element);
  if (shape.empty()) return element;
  return wk::ValueType::get(element.getContext(), element,
                            DenseI64ArrayAttr::get(element.getContext(), shape),
                            DenseI64ArrayAttr::get(element.getContext(), axes));
}

FailureOr<Type> pointwiseValueType(Type element, ValueRange operands) {
  SmallVector<int64_t> shape, axes;
  for (Value operand : operands) {
    auto value = dyn_cast<wk::ValueType>(operand.getType());
    if (!value) continue;
    for (auto [axis, extent] : llvm::zip(value.getAxisIds().asArrayRef(),
                                        value.getShape().asArrayRef())) {
      auto found = llvm::find(axes, axis);
      if (found == axes.end()) {
        axes.push_back(axis);
        shape.push_back(extent);
      } else {
        int64_t &previous = shape[found - axes.begin()];
        if (previous == 1) previous = extent;
        else if (extent != 1 && previous != extent) return failure();
      }
    }
  }
  return nativeValueType(element, shape, axes);
}

FailureOr<Value> createBinaryValue(OpBuilder &builder, Location location,
                                   Value lhs, Value rhs, StringRef kind) {
  if (elementType(lhs.getType()) != elementType(rhs.getType()))
    return emitError(location, "Weft binary operands have different element types"),
           failure();
  auto type = pointwiseValueType(elementType(lhs.getType()), {lhs, rhs});
  if (failed(type))
    return emitError(location, "Weft pointwise operands have incompatible axis/extent relations"),
           failure();
  return Value(builder.create<wk::BinaryOp>(location, *type, lhs, rhs, kind));
}

namespace {

class ScalarConversion {
public:
  ScalarConversion(OpBuilder &builder, const ScalarOperation &scalar,
                   ValueRange operands)
      : b(builder), scalar(scalar), inputs(operands), loc(scalar.operation->getLoc()) {}

  FailureOr<Value> convert();

private:
  FailureOr<Value> unsupported(StringRef reason) {
    return scalar.operation->emitError("Weft numerical conversion unsupported: ")
               << reason,
           failure();
  }

  Value constant(Type type, int64_t value) {
    if (type.isIndex()) return b.create<arith::ConstantIndexOp>(loc, value);
    return b.create<wk::ConstantOp>(loc, type, b.getIntegerAttr(type, value));
  }

  Value convertType(Value value, Type element) {
    Type target = withElementType(value.getType(), element);
    if (target == value.getType()) return value;
    return b.create<wk::CastOp>(loc, target, value);
  }

  FailureOr<Value> binary(Value lhs, Value rhs, StringRef kind) {
    return createBinaryValue(b, loc, lhs, rhs, kind);
  }

  FailureOr<Value> compare(Value lhs, Value rhs, StringRef predicate) {
    if (elementType(lhs.getType()) != elementType(rhs.getType()))
      return unsupported("comparison operands have different element types");
    auto type = pointwiseValueType(b.getI1Type(), {lhs, rhs});
    if (failed(type)) return unsupported("comparison domains disagree");
    return Value(b.create<wk::CompareOp>(loc, *type, lhs, rhs, predicate));
  }

  FailureOr<Value> select(Value predicate, Value lhs, Value rhs) {
    if (elementType(lhs.getType()) != elementType(rhs.getType()))
      return unsupported("select branches have different element types");
    auto type = pointwiseValueType(elementType(lhs.getType()),
                                   {lhs, rhs, predicate});
    if (failed(type)) return unsupported("select operand domains disagree");
    return Value(b.create<wk::SelectOp>(loc, *type, predicate, lhs, rhs));
  }

  FailureOr<Value> integerInput(Value value, bool unsignedInput) {
    Type source = elementType(value.getType());
    if (source.isInteger(1)) {
      // arith signed i1 interprets true as -1; a Weft predicate is a boolean.
      Type carrier = IntegerType::get(b.getContext(), 8,
          unsignedInput ? IntegerType::Unsigned : IntegerType::Signed);
      return select(value, constant(carrier, unsignedInput ? 1 : -1),
                     constant(carrier, 0));
    }
    Type target = IntegerType::get(b.getContext(), integerWidth(source),
        unsignedInput ? IntegerType::Unsigned : IntegerType::Signed);
    return convertType(value, target);
  }

  FailureOr<Value> truncateInteger(Value value, Type destination) {
    unsigned width = integerWidth(destination);
    auto input = integerInput(value, /*unsignedInput=*/true);
    if (failed(input)) return failure();
    value = *input;
    Type source = elementType(value.getType());
    unsigned sourceWidth = integerWidth(source);
    if (width < sourceWidth) {
      auto mask = llvm::APInt::getLowBitsSet(sourceWidth, width);
      Value literal = b.create<wk::ConstantOp>(loc, source,
                                               b.getIntegerAttr(source, mask));
      auto masked = binary(value, literal, "and");
      if (failed(masked)) return failure();
      value = *masked;
    }
    if (width == 1)
      return compare(value, constant(elementType(value.getType()), 0), "ne");
    // The native narrowing instruction halves SEW. Masking before narrowing
    // also keeps scalar/tuple realization within the destination's range.
    while (sourceWidth > width) {
      sourceWidth /= 2;
      Type narrowed = IntegerType::get(b.getContext(), sourceWidth,
                                       IntegerType::Unsigned);
      value = b.create<wk::NarrowOp>(loc,
          withElementType(value.getType(), narrowed), value, "rtz", false);
    }
    return convertType(value, nativeScalarType(destination));
  }

  FailureOr<Value> finishInteger(Value value) {
    Type target = nativeScalarType(scalar.type());
    if (target.isInteger(1)) return truncateInteger(value, target);
    return convertType(value, target);
  }

  FailureOr<Value> integerBinary(StringRef kind, bool unsignedInputs,
                                 bool wrap = false) {
    auto lhs = integerInput(inputs[0], unsignedInputs);
    auto rhs = integerInput(inputs[1], unsignedInputs);
    if (failed(lhs) || failed(rhs)) return failure();
    unsigned width = integerWidth(scalar.type());
    if (wrap && kind == "mul" && width == 16) {
      // C promotes u16 operands to signed int, whose product can overflow.
      Type wide = IntegerType::get(b.getContext(), 32, IntegerType::Unsigned);
      *lhs = b.create<wk::WidenOp>(loc, withElementType(lhs->getType(), wide), *lhs);
      *rhs = b.create<wk::WidenOp>(loc, withElementType(rhs->getType(), wide), *rhs);
    }
    auto result = binary(*lhs, *rhs, kind);
    if (failed(result)) return failure();
    if (wrap && integerWidth(result->getType()) > width)
      return truncateInteger(*result, scalar.type());
    return finishInteger(*result);
  }

  FailureOr<Value> roundedDivision(bool floor, bool unsignedInputs);
  FailureOr<Value> comparison();
  FailureOr<Value> cast();

  OpBuilder &b;
  const ScalarOperation &scalar;
  ValueRange inputs;
  Location loc;
};

FailureOr<Value> ScalarConversion::roundedDivision(bool floor,
                                                   bool unsignedInputs) {
  auto lhs = integerInput(inputs[0], unsignedInputs);
  auto rhs = integerInput(inputs[1], unsignedInputs);
  if (failed(lhs) || failed(rhs)) return failure();
  auto quotient = binary(*lhs, *rhs, "div");
  auto remainder = binary(*lhs, *rhs, "mod");
  if (failed(quotient) || failed(remainder)) return failure();
  Type type = elementType(lhs->getType());
  auto nonzero = compare(*remainder, constant(type, 0), "ne");
  if (failed(nonzero)) return failure();
  Value adjust = *nonzero;
  if (!unsignedInputs) {
    auto leftNegative = compare(*lhs, constant(type, 0), "lt");
    auto rightNegative = compare(*rhs, constant(type, 0), "lt");
    if (failed(leftNegative) || failed(rightNegative)) return failure();
    auto signs = compare(*leftNegative, *rightNegative, floor ? "ne" : "eq");
    if (failed(signs)) return failure();
    auto active = binary(adjust, *signs, "and");
    if (failed(active)) return failure();
    adjust = *active;
  }
  auto increment = select(adjust, constant(type, 1), constant(type, 0));
  if (failed(increment)) return failure();
  auto result = binary(*quotient, *increment, floor ? "sub" : "add");
  return succeeded(result) ? finishInteger(*result) : FailureOr<Value>(failure());
}

FailureOr<Value> ScalarConversion::comparison() {
  auto semantics = scalar.comparison;
  Value lhs = inputs[0], rhs = inputs[1];
  if (semantics.nan == ScalarNaN::None) {
    auto left = integerInput(lhs, semantics.unsignedInteger);
    auto right = integerInput(rhs, semantics.unsignedInteger);
    if (failed(left) || failed(right)) return failure();
    return compare(*left, *right, scalarComparisonMethod(semantics.relation));
  }
  if (semantics.nan == ScalarNaN::AlwaysFalse ||
      semantics.nan == ScalarNaN::AlwaysTrue) {
    auto type = pointwiseValueType(b.getI1Type(), inputs);
    if (failed(type)) return unsupported("comparison domains disagree");
    Value literal = constant(b.getI1Type(), semantics.nan == ScalarNaN::AlwaysTrue);
    return isa<wk::ValueType>(*type)
               ? Value(b.create<wk::NewOp>(loc, *type, literal)) : literal;
  }
  // Native relations already have IEEE ordered semantics except `ne`, whose
  // result is true for NaN. Preserve that direct surface when it is exact.
  if (semantics.relationResult &&
      ((semantics.nan == ScalarNaN::Ordered &&
        semantics.relation != ScalarRelation::NotEqual) ||
       (semantics.nan == ScalarNaN::Unordered &&
        semantics.relation == ScalarRelation::NotEqual)))
    return compare(lhs, rhs, scalarComparisonMethod(semantics.relation));
  auto leftNaN = compare(lhs, lhs, "ne");
  auto rightNaN = compare(rhs, rhs, "ne");
  if (failed(leftNaN) || failed(rightNaN)) return failure();
  auto unordered = binary(*leftNaN, *rightNaN, "or");
  if (failed(unordered)) return failure();
  Value classification = *unordered;
  if (semantics.nan == ScalarNaN::Ordered) {
    auto ordered = binary(classification, constant(b.getI1Type(), 1), "xor");
    if (failed(ordered)) return failure();
    classification = *ordered;
  }
  if (!semantics.relationResult) return classification;
  auto relation = compare(lhs, rhs, scalarComparisonMethod(semantics.relation));
  if (failed(relation)) return failure();
  return binary(*relation, classification,
                 semantics.nan == ScalarNaN::Ordered ? "and" : "or");
}

FailureOr<Value> ScalarConversion::cast() {
  Value input = inputs.front();
  Type destination = nativeScalarType(scalar.type());
  ScalarCast kind = *scalar.cast;
  bool unsignedInput = kind == ScalarCast::ExtendUnsigned ||
      kind == ScalarCast::UnsignedToFloat || kind == ScalarCast::IndexUnsigned;
  if (scalar.inputType().isIntOrIndex()) {
    auto interpreted = integerInput(input, unsignedInput);
    if (failed(interpreted)) return failure();
    input = *interpreted;
    if (destination.isIntOrIndex()) {
      unsigned from = integerWidth(input.getType());
      unsigned to = integerWidth(destination);
      if (from > to) return truncateInteger(input, destination);
      if (to > from) {
        Type target = IntegerType::get(b.getContext(), to,
            unsignedInput ? IntegerType::Unsigned : IntegerType::Signed);
        input = b.create<wk::WidenOp>(loc,
            withElementType(input.getType(), target), input);
      }
      return convertType(input, destination);
    }
    if (destination.isF32() && integerWidth(input.getType()) < 32) {
      Type wide = IntegerType::get(b.getContext(), 32,
          unsignedInput ? IntegerType::Unsigned : IntegerType::Signed);
      input = b.create<wk::WidenOp>(loc,
          withElementType(input.getType(), wide), input);
    }
    return convertType(input, destination);
  }
  if (kind == ScalarCast::FloatToSigned || kind == ScalarCast::FloatToUnsigned) {
    Type target = IntegerType::get(b.getContext(),
        std::max(8u, integerWidth(destination)),
        kind == ScalarCast::FloatToUnsigned ? IntegerType::Unsigned
                                           : IntegerType::Signed);
    if (elementType(input.getType()).getIntOrFloatBitWidth() >
        target.getIntOrFloatBitWidth())
      input = b.create<wk::NarrowOp>(loc,
          withElementType(input.getType(), target), input, "rtz", false);
    else input = convertType(input, target);
    return destination.isInteger(1) ? truncateInteger(input, destination)
                                    : FailureOr<Value>(convertType(input, destination));
  }
  if (kind == ScalarCast::TruncateFloat)
    return Value(b.create<wk::NarrowOp>(loc,
        withElementType(input.getType(), destination), input, "rne", false));
  return convertType(input, destination);
}

FailureOr<Value> ScalarConversion::convert() {
  using K = ScalarKind;
  if (scalar.kind == K::Constant) {
    Type type = nativeScalarType(scalar.type());
    if (type == scalar.type()) return b.clone(*scalar.operation)->getResult(0);
    auto integer = dyn_cast<IntegerAttr>(scalar.constant);
    if (!integer) return unsupported("constant has no native scalar value");
    return Value(b.create<wk::ConstantOp>(loc, type,
                                         b.getIntegerAttr(type, integer.getValue())));
  }
  switch (scalar.kind) {
  case K::Compare: return comparison();
  case K::Select: return select(inputs[0], inputs[1], inputs[2]);
  case K::Cast: return cast();
  case K::Bitcast: {
    Type target = withElementType(inputs[0].getType(), nativeScalarType(scalar.type()));
    if (target == inputs[0].getType()) return inputs[0];
    return Value(b.create<wk::BitcastOp>(loc, target, inputs[0]));
  }
  case K::Add: case K::Subtract: case K::Multiply: {
    StringRef kind = scalar.kind == K::Add ? "add"
                     : scalar.kind == K::Subtract ? "sub" : "mul";
    if (scalar.type().isInteger(1))
      return binary(inputs[0], inputs[1], scalar.kind == K::Multiply ? "and" : "xor");
    if (scalar.type().isIntOrIndex()) return integerBinary(kind, true, true);
    return binary(inputs[0], inputs[1], kind);
  }
  case K::DivideFloat: return binary(inputs[0], inputs[1], "div");
  case K::DivideSigned: return integerBinary("div", false);
  case K::DivideUnsigned: return integerBinary("div", true);
  case K::FloorDivideSigned: return roundedDivision(true, false);
  case K::CeilDivideSigned: return roundedDivision(false, false);
  case K::CeilDivideUnsigned: return roundedDivision(false, true);
  case K::RemainderSigned: {
    auto lhs = integerInput(inputs[0], false);
    auto rhs = integerInput(inputs[1], false);
    if (failed(lhs) || failed(rhs)) return failure();
    Type type = elementType(rhs->getType());
    auto negativeOne = compare(*rhs, constant(type, -1), "eq");
    if (failed(negativeOne)) return failure();
    auto divisor = select(*negativeOne, constant(type, 1), *rhs);
    if (failed(divisor)) return failure();
    auto result = binary(*lhs, *divisor, "mod");
    return succeeded(result) ? finishInteger(*result) : FailureOr<Value>(failure());
  }
  case K::RemainderUnsigned: return integerBinary("mod", true);
  case K::ShiftLeft: return integerBinary("shl", true, true);
  case K::ShiftRightSigned: return integerBinary("shr", false);
  case K::ShiftRightUnsigned: return integerBinary("shr", true);
  case K::And: case K::Or: case K::Xor:
    return binary(inputs[0], inputs[1], scalar.kind == K::And ? "and"
                  : scalar.kind == K::Or ? "or" : "xor");
  case K::MinimumSigned: return integerBinary("min", false);
  case K::MaximumSigned: return integerBinary("max", false);
  case K::MinimumUnsigned: return integerBinary("min", true);
  case K::MaximumUnsigned: return integerBinary("max", true);
  case K::MinimumFloat: return binary(inputs[0], inputs[1], "minimum");
  case K::MaximumFloat: return binary(inputs[0], inputs[1], "maximum");
  case K::MinimumNumber: return binary(inputs[0], inputs[1], "min");
  case K::MaximumNumber: return binary(inputs[0], inputs[1], "max");
  case K::AbsInteger: {
    auto input = integerInput(inputs[0], false);
    if (failed(input)) return failure();
    auto negative = compare(*input, constant(elementType(input->getType()), 0), "lt");
    auto bits = integerInput(inputs[0], true);
    if (failed(negative) || failed(bits)) return failure();
    auto opposite = binary(constant(elementType(bits->getType()), 0), *bits, "sub");
    if (failed(opposite)) return failure();
    auto result = select(*negative, *opposite, *bits);
    return succeeded(result) ? finishInteger(*result) : FailureOr<Value>(failure());
  }
  case K::Negate: case K::AbsFloat: case K::Exp: case K::Exp2: case K::Log:
  case K::Sqrt: case K::Rsqrt: {
    StringRef kind = scalar.kind == K::Negate ? "neg"
                     : scalar.kind == K::AbsFloat ? "abs"
                     : scalar.kind == K::Exp ? "exp"
                     : scalar.kind == K::Exp2 ? "exp2"
                     : scalar.kind == K::Log ? "log"
                     : scalar.kind == K::Sqrt ? "sqrt" : "rsqrt";
    auto result = b.create<wk::UnaryOp>(loc, inputs[0].getType(), inputs[0], kind);
    if (auto math = dyn_cast<arith::ArithFastMathInterface>(scalar.operation))
      if ((math.getFastMathFlagsAttr().getValue() & arith::FastMathFlags::afn) !=
          arith::FastMathFlags::none)
        result->setAttr("approximate", b.getBoolAttr(true));
    return Value(result);
  }
  case K::Constant: llvm_unreachable("constant handled before dispatch");
  case K::Tanh: case K::Sin: case K::Cos:
  case K::Floor: case K::Erf: case K::Power: case K::Fma:
    return unsupported("operation has no equivalent canonical Weft primitive");
  }
  llvm_unreachable("invalid scalar semantic kind");
}

} // namespace

FailureOr<Value> lowerScalarValue(OpBuilder &builder, Operation *operation,
                                  ValueRange operands) {
  auto scalar = ScalarOperation::read(operation);
  if (failed(scalar)) return failure();
  if (operands.size() != operation->getNumOperands())
    return operation->emitError("Weft scalar operand binding is incomplete"), failure();
  if (!isa<IntegerType, IndexType, FloatType>(scalar->type()))
    return operation->emitError("Weft scalar lifting requires a scalar source result"), failure();
  for (Type type : llvm::concat<Type>(operation->getOperandTypes(),
                                      operation->getResultTypes()))
    if (auto integer = dyn_cast<IntegerType>(type);
        integer && !llvm::is_contained(ArrayRef<unsigned>{1, 8, 16, 32, 64},
                                       integer.getWidth()))
      return operation->emitError("integer width has no supported Weft numeric carrier: ")
                 << integer.getWidth(),
             failure();
  return ScalarConversion(builder, *scalar, operands).convert();
}

} // namespace intent::weft_provider
