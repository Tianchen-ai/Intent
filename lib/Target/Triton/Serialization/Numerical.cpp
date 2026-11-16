#include "Intent/Target/Triton/Serialization/Numerical.h"
#include "Intent/Target/Triton/IR/Target.h"

#include "Intent/Dialect/Block/IR/Program.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <type_traits>

using namespace mlir;

namespace intent::triton {
namespace {

using Dependencies = block::PythonEmitter::Emitters::Dependencies;

struct Translation {
  std::string expression;
  Dependencies dependencies;
};

Type elementType(Type type) { return block::PythonEmitter::elementType(type); }

FailureOr<std::string> scalarType(Operation *operation, Type type) {
  type = elementType(type);
  if (auto integer = dyn_cast<IntegerType>(type)) {
    unsigned width = integer.getWidth();
    if (width != 1 && width != 8 && width != 16 && width != 32 && width != 64)
      return operation->emitOpError("has no Triton numerical integer type"),
             failure();
  }
  std::string result = block::pythonScalarType(
      type, {"tl.", "int1", "float64", "float8e4nv", "float8e5"});
  if (result.empty())
    return operation->emitOpError("has no Triton numerical scalar type"),
           failure();
  return result;
}

std::string castValue(StringRef value, StringRef type, bool bitcast = false) {
  return "tl.cast(" + value.str() + ", " + type.str() +
         (bitcast ? ", bitcast=True)" : ")");
}

Translation direct(std::string expression) {
  return {std::move(expression), {}};
}

Translation library(std::string expression) {
  return {std::move(expression),
          {"from triton.language.extra import libdevice"}};
}

FailureOr<Translation> approximate(Operation *operation, ApproximateMath kind,
                                   ArrayRef<std::string> operands, bool ftz) {
  Attribute target = targetOf(operation);
  auto interface = targetInterface(target);
  if (!interface)
    return operation->emitOpError("requires a registered Triton numerical target"), failure();
  auto expression = interface->approximateMath(operation, target, kind, operands, ftz);
  if (failed(expression)) return failure();
  return library(std::move(*expression));
}

struct SignedDivision {
  explicit SignedDivision(StringRef lhs, StringRef rhs)
      : quotient("(" + lhs.str() + " // " + rhs.str() + ")"),
        remainder("(" + lhs.str() + " % " + rhs.str() + ")"),
        nonzero("(" + remainder + " != 0)"),
        adjustment("(" + nonzero + " & ((" + remainder + " < 0) != (" +
                   rhs.str() + " < 0)))") {}

  // Triton's runtime // truncates; Python constexpr already floors and has a
  // divisor-sign remainder, making the same correction zero.
  std::string floor() const { return "(" + quotient + " - " + adjustment + ")"; }
  std::string ceil() const { return "(" + floor() + " + " + nonzero + ")"; }
  std::string modulo(StringRef rhs) const {
    return "(" + remainder + " + " + adjustment + " * " + rhs.str() + ")";
  }

  std::string quotient;
  std::string remainder;
  std::string nonzero;
  std::string adjustment;
};

FailureOr<Translation> translate(arith::ConstantOp operation,
                                ArrayRef<std::string>, bool) {
  Attribute attribute = operation.getValue();
  if (!isa<IntegerAttr, FloatAttr>(attribute))
    return operation.emitOpError(
               "Python Block constants require a scalar integer or floating value"),
           failure();
  auto type = scalarType(operation, operation.getType());
  if (failed(type))
    return failure();
  std::string literal = block::pythonLiteral(attribute);
  auto floating = dyn_cast<FloatAttr>(attribute);
  if (!floating)
    return direct(std::move(literal));
  if (isa<Float8E4M3FNType, Float8E5M2Type>(floating.getType())) {
    uint64_t encoding = floating.getValue().bitcastToAPInt().getZExtValue();
    return direct(castValue(
        "tl.full((), " + std::to_string(encoding) + ", tl.uint8)", *type, true));
  }
  std::string expression = "tl.full((), " + literal + ", " + *type + ")";
  // Triton's scalar constructor canonicalizes either zero sign to +0.
  if (floating.getValue().isNegZero())
    expression = "(-" + expression + ")";
  return direct(std::move(expression));
}

FailureOr<Translation> libraryMath(Operation *operation, StringRef name,
                                  ArrayRef<std::string> operands) {
  auto element =
      dyn_cast<FloatType>(elementType(operation->getResult(0).getType()));
  if (!element)
    return operation->emitOpError(
               "Triton mathematical library lowering requires floating values"),
           failure();
  auto resultType = scalarType(operation, element);
  if (failed(resultType))
    return failure();
  StringRef computation = *resultType;
  if (element.getWidth() < 32)
    computation = "tl.float32";
  std::string expression = "libdevice." + name.str() + "(";
  for (auto [index, operand] : llvm::enumerate(operands)) {
    if (index)
      expression += ", ";
    expression += castValue(operand, computation);
  }
  expression += ")";
  if (element.getWidth() < 32) {
    auto narrowed = numericalCast(
        operation, Float32Type::get(operation->getContext()), element,
        expression, false);
    if (failed(narrowed))
      return failure();
    expression = std::move(*narrowed);
  }
  return library(std::move(expression));
}

FailureOr<Translation> translate(block::BinaryOp operation,
                                ArrayRef<std::string> operands,
                                bool compileTime) {
  const std::string &lhs = operands[0];
  const std::string &rhs = operands[1];
  auto infix = [&](StringRef spelling) {
    return "(" + lhs + " " + spelling.str() + " " + rhs + ")";
  };
  auto minMax = [&](StringRef name, StringRef nan) {
    return direct("tl." + name.str() + "(" + lhs + ", " + rhs +
                  ", propagate_nan=tl.PropagateNan." + nan.str() + ")");
  };
  if (operation.getStrictRounding()) {
    switch (operation.getOperatorKind()) {
    case BinaryOperator::Add: return libraryMath(operation, "add_rn", operands);
    case BinaryOperator::Subtract: return libraryMath(operation, "sub_rn", operands);
    case BinaryOperator::Multiply: return libraryMath(operation, "mul_rn", operands);
    default:
      return operation.emitOpError("unsupported strict-rounding binary operation"), failure();
    }
  }
  switch (operation.getOperatorKind()) {
  case BinaryOperator::Add:
    return direct(infix("+"));
  case BinaryOperator::Subtract:
    return direct(infix("-"));
  case BinaryOperator::Multiply:
    return direct(infix("*"));
  case BinaryOperator::TrueDivide:
    if (operation.getApproximate())
      return approximate(operation, ApproximateMath::Divide, operands, operation.getFlushToZero());
    return libraryMath(operation, "div_rn", operands);
  case BinaryOperator::FloorDivide:
  case BinaryOperator::Remainder: {
    auto integer =
        dyn_cast<IntegerType>(elementType(operation.getResult().getType()));
    bool remainder = operation.getOperatorKind() == BinaryOperator::Remainder;
    if (integer && integer.isUnsigned())
      return direct(infix(remainder ? "%" : "//"));
    SignedDivision division(lhs, rhs);
    return direct(remainder ? division.modulo(rhs) : division.floor());
  }
  case BinaryOperator::Power:
    return libraryMath(operation, "pow", operands);
  case BinaryOperator::Maximum:
    if (compileTime && operation.getResult().getType().isIndex())
      return direct("max(" + lhs + ", " + rhs + ")");
    return minMax("maximum", "ALL");
  case BinaryOperator::Minimum:
    if (compileTime && operation.getResult().getType().isIndex())
      return direct("min(" + lhs + ", " + rhs + ")");
    return minMax("minimum", "ALL");
  case BinaryOperator::MaximumNum:
    return minMax("maximum", "NONE");
  case BinaryOperator::MinimumNum:
    return minMax("minimum", "NONE");
  case BinaryOperator::LogicalAnd:
  case BinaryOperator::BitwiseAnd:
    return direct(infix("&"));
  case BinaryOperator::LogicalOr:
  case BinaryOperator::BitwiseOr:
    return direct(infix("|"));
  case BinaryOperator::BitwiseXor:
    return direct(infix("^"));
  case BinaryOperator::LeftShift:
    return direct(infix("<<"));
  case BinaryOperator::RightShift:
    return direct(infix(">>"));
  }
  llvm_unreachable("unhandled Intent binary operator");
}

FailureOr<Translation> translate(block::UnaryOp operation,
                                ArrayRef<std::string> operands, bool) {
  const std::string &input = operands.front();
  auto native = [&](StringRef name) {
    return direct("tl." + name.str() + "(" + input + ")");
  };
  switch (operation.getOperatorKind()) {
  case UnaryOperator::Negate:
    return direct("(-" + input + ")");
  case UnaryOperator::Not:
    return direct("(~" + input + ")");
  case UnaryOperator::Exp:
    return libraryMath(operation, "exp", operands);
  case UnaryOperator::Exp2:
    if (operation.getApproximate())
      return approximate(operation, ApproximateMath::Exp2, operands, operation.getFlushToZero());
    return libraryMath(operation, "exp2", operands);
  case UnaryOperator::Log:
    return libraryMath(operation, "log", operands);
  case UnaryOperator::Log1p:
    return libraryMath(operation, "log1p", operands);
  case UnaryOperator::Lgamma:
    return libraryMath(operation, "lgamma", operands);
  case UnaryOperator::Sin:
    return libraryMath(operation, "sin", operands);
  case UnaryOperator::Asin:
    return libraryMath(operation, "asin", operands);
  case UnaryOperator::Cos:
    return libraryMath(operation, "cos", operands);
  case UnaryOperator::Floor:
    return native("floor");
  case UnaryOperator::Erf:
    return libraryMath(operation, "erf", operands);
  case UnaryOperator::Erfc:
    return libraryMath(operation, "erfc", operands);
  case UnaryOperator::I0:
    return libraryMath(operation, "cyl_bessel_i0", operands);
  case UnaryOperator::Rsqrt:
    return libraryMath(operation, "rsqrt", operands);
  case UnaryOperator::Sigmoid:
    return native("sigmoid");
  case UnaryOperator::Tanh:
    if (operation.getApproximate())
      return approximate(operation, ApproximateMath::Tanh, operands, false);
    return libraryMath(operation, "tanh", operands);
  case UnaryOperator::Abs:
    return native("abs");
  case UnaryOperator::Sqrt:
    return libraryMath(operation, "sqrt_rn", operands);
  }
  llvm_unreachable("unhandled Intent unary operator");
}

FailureOr<Translation> translate(block::CastOp operation,
                                ArrayRef<std::string> operands, bool) {
  auto expression = numericalCast(operation, operation.getValue().getType(),
                                 operation.getResult().getType(),
                                 operands.front(), false);
  if (failed(expression))
    return failure();
  return direct(std::move(*expression));
}

FailureOr<Translation> translate(block::BitcastOp operation,
                                ArrayRef<std::string> operands, bool) {
  auto expression = numericalCast(operation, operation.getValue().getType(),
                                 operation.getResult().getType(),
                                 operands.front(), true);
  if (failed(expression))
    return failure();
  return direct(std::move(*expression));
}

FailureOr<Translation> translate(block::CompareOp operation,
                                ArrayRef<std::string> operands, bool) {
  StringRef token;
  switch (operation.getPredicate()) {
  case ComparePredicate::Eq:
    token = "==";
    break;
  case ComparePredicate::Ne:
    token = "!=";
    break;
  case ComparePredicate::Lt:
    token = "<";
    break;
  case ComparePredicate::Le:
    token = "<=";
    break;
  case ComparePredicate::Gt:
    token = ">";
    break;
  case ComparePredicate::Ge:
    token = ">=";
    break;
  }
  return direct("(" + operands[0] + " " + token.str() + " " + operands[1] + ")");
}

FailureOr<Translation> translate(block::SelectOp, ArrayRef<std::string> operands,
                                bool) {
  return direct("tl.where(" + operands[0] + ", " + operands[1] + ", " +
                operands[2] + ")");
}

template <typename Op> FailureOr<Translation> inspect(Op operation) {
  SmallVector<std::string> operands(operation->getNumOperands(), "value");
  return translate(operation, operands, false);
}

template <typename Op> void addNumerical(block::PythonEmitter::Emitters &emitters) {
  emitters.add<Op>(
      [](Op operation) { return failure(failed(inspect(operation))); },
      [](Op operation, block::PythonEmitter &emitter) -> LogicalResult {
        SmallVector<std::string> operands;
        for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
          bool controlValue = std::is_same_v<Op, block::SelectOp> && index != 0;
          operands.push_back(controlValue ? emitter.controlValueString(operand)
                                          : emitter.valueString(operand));
        }
        bool compileTime = false;
        if constexpr (std::is_same_v<Op, block::BinaryOp> ||
                      std::is_same_v<Op, block::CompareOp>) {
          Type type = operation.getResult().getType();
          compileTime = (type.isIndex() || type.isInteger(1)) &&
                        llvm::all_of(operation->getOperands(), [&](Value value) {
                          return emitter.isConstexprValue(value);
                        });
        }
        auto translated = translate(operation, operands, compileTime);
        if (failed(translated))
          return failure();
        emitter.assign(operation.getResult(), translated->expression,
                       compileTime);
        return success();
      },
      [](Op operation) -> FailureOr<Dependencies> {
        auto translated = inspect(operation);
        if (failed(translated))
          return failure();
        return std::move(translated->dependencies);
      });
}

} // namespace

std::string integerDivision(block::PhysicalExprKind kind, StringRef lhs,
                            StringRef rhs) {
  SignedDivision division(lhs, rhs);
  switch (kind) {
  case block::PhysicalExprKind::FloorDiv: return division.floor();
  case block::PhysicalExprKind::CeilDiv: return division.ceil();
  default: llvm_unreachable("expected a physical integer division");
  }
}

FailureOr<std::string> numericalCast(Operation *diagnostic, Type source,
                                    Type target, StringRef value, bool bitcast) {
  source = elementType(source);
  target = elementType(target);
  auto type = scalarType(diagnostic, target);
  if (failed(type) || failed(scalarType(diagnostic, source)))
    return failure();
  if (bitcast)
    return castValue(value, *type, true);

  std::string input = value.str();
  bool sourceFP8 = isa<Float8E4M3FNType, Float8E5M2Type>(source);
  bool targetFP8 = isa<Float8E4M3FNType, Float8E5M2Type>(target);
  // Triton's integer/FP8 conversions use the ordinary floating bridge. Every
  // non-overflowing integer in either FP8 format is exactly representable in
  // f32; E5M2 overflow is decided from the original integer below.
  if ((source.isIntOrIndex() && targetFP8) ||
      (sourceFP8 && target.isIntOrIndex()))
    input = castValue(input, "tl.float32");
  std::string converted = castValue(input, *type);
  if (!isa<Float8E5M2Type>(target) || sourceFP8)
    return converted;

  // NVIDIA's FP8 conversion saturates finite values, whereas E5M2's ordinary
  // cast overflows to infinity. At the RN-even midpoint 61440 the next exponent
  // wins. Select the encoding before bitcasting to keep the FP8 result type.
  std::string overflow;
  std::string infinity = "tl.full((), 124, tl.uint8)";
  auto integer = dyn_cast<IntegerType>(source);
  if (source.isIntOrIndex()) {
    bool isUnsigned = integer && integer.isUnsigned();
    unsigned width = integer ? integer.getWidth() : 64;
    if (width < 16 || (!isUnsigned && width == 16))
      return converted;
    overflow = "(" + value.str() + " >= 61440)";
    if (!isUnsigned)
      overflow = "(" + overflow + " | (" + value.str() + " <= -61440))";
    if (!isUnsigned)
      infinity = "tl.where(" + value.str() +
                 " < 0, tl.full((), 252, tl.uint8), " + infinity + ")";
  } else {
    overflow = "(tl.abs(" + value.str() + ") >= 61440.0)";
    infinity = "tl.where(" + value.str() +
               " < 0, tl.full((), 252, tl.uint8), " + infinity + ")";
  }
  std::string bits = castValue(converted, "tl.uint8", true);
  return castValue("tl.where(" + overflow + ", " + infinity + ", " + bits + ")",
                   *type, true);
}

void addNumericalOperations(block::PythonEmitter::Emitters &emitters) {
  emitters.replace<arith::ConstantOp>(
      [](arith::ConstantOp operation) {
        return failure(failed(inspect(operation)));
      },
      emitNumericalConstant,
      [](arith::ConstantOp operation) -> FailureOr<Dependencies> {
        auto translated = inspect(operation);
        if (failed(translated))
          return failure();
        return std::move(translated->dependencies);
      });
  addNumerical<block::BinaryOp>(emitters);
  addNumerical<block::UnaryOp>(emitters);
  addNumerical<block::CastOp>(emitters);
  addNumerical<block::BitcastOp>(emitters);
  addNumerical<block::CompareOp>(emitters);
  addNumerical<block::SelectOp>(emitters);
}

LogicalResult emitNumericalConstant(arith::ConstantOp operation,
                                   block::PythonEmitter &emitter) {
  auto translated = inspect(operation);
  if (failed(translated))
    return failure();
  if (isa<IntegerAttr>(operation.getValue()))
    emitter.bindConstexpr(operation.getResult(), translated->expression);
  else
    emitter.assign(operation.getResult(), translated->expression);
  return success();
}

} // namespace intent::triton
