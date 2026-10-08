#include "Intent/Target/CuTile/Serialization/Numerical.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace intent::cutile {
namespace {

using Emitter = gpu::PythonEmitter;
using Dependencies = Emitter::Emitters::Dependencies;

struct Translation {
  std::string expression;
  Dependencies dependencies;
  bool propagatesConstexpr = false;
};

std::string typeName(Type type) {
  return gpu::pythonScalarType(
      Emitter::elementType(type),
      {"ct.", "bool_", "float64", "float8_e4m3fn", "float8_e5m2"});
}

std::string castExpression(StringRef value, Type source, Type result,
                           bool bitcast) {
  if (Emitter::elementType(source) == Emitter::elementType(result))
    bitcast = false;
  return std::string(bitcast ? "ct.bitcast(" : "ct.astype(") + value.str() +
         ", " + typeName(result) + ")";
}

std::string selectExpression(ArrayRef<std::string> operands) {
  return "ct.where(" + operands[0] + ", " + operands[1] + ", " +
         operands[2] + ")";
}

FailureOr<Translation> translate(gpu::BinaryOp operation,
                                 ArrayRef<std::string> operands) {
  if (operation.getStrictRounding())
    return operation.emitOpError("strict f32 arithmetic must be legalized before cuTile source emission"), failure();
  Type element = Emitter::elementType(operation.getLhs().getType());
  auto expression = [&](std::string value) -> FailureOr<Translation> {
    return Translation{std::move(value), {}, true};
  };
  auto arithmetic = [&](std::string value) -> FailureOr<Translation> {
    if (element.isInteger(1) ||
        isa<Float8E4M3FNType, Float8E5M2Type>(element))
      return operation.emitOpError(
          "cuTile native binary arithmetic does not support this element type: ")
                 << element,
             failure();
    return expression(std::move(value));
  };
  auto infix = [&](StringRef token) {
    return expression("(" + operands[0] + " " + token.str() + " " +
                      operands[1] + ")");
  };
  auto arithmeticInfix = [&](StringRef token) {
    return arithmetic("(" + operands[0] + " " + token.str() + " " +
                      operands[1] + ")");
  };
  auto minmax = [&](StringRef function, bool propagateNaN) {
    std::string value = function.str() + "(" + operands[0] + ", " +
                        operands[1] + ")";
    if (propagateNaN &&
        !Emitter::elementType(operation.getResult().getType()).isIntOrIndex())
      value = "ct.where(ct.isnan(" + operands[0] + "), " + operands[0] +
              ", ct.where(ct.isnan(" + operands[1] + "), " + operands[1] +
              ", " + value + "))";
    return arithmetic(std::move(value));
  };
  switch (operation.getOperatorKind()) {
  case BinaryOperator::Add: return arithmeticInfix("+");
  case BinaryOperator::Subtract: return arithmeticInfix("-");
  case BinaryOperator::Multiply: return arithmeticInfix("*");
  case BinaryOperator::TrueDivide:
    if (operation.getApproximate())
      return arithmetic("ct.truediv(" + operands[0] + ", " + operands[1] +
          ", rounding_mode=ct.RoundingMode.APPROX, flush_to_zero=" +
          (operation.getFlushToZero() ? "True" : "False") + ")");
    return arithmeticInfix("/");
  case BinaryOperator::FloorDivide: return arithmeticInfix("//");
  case BinaryOperator::Remainder: return arithmeticInfix("%");
  case BinaryOperator::Power: return arithmeticInfix("**");
  case BinaryOperator::MaximumNum: return minmax("ct.maximum", false);
  case BinaryOperator::MinimumNum: return minmax("ct.minimum", false);
  case BinaryOperator::Maximum: return minmax("ct.maximum", true);
  case BinaryOperator::Minimum: return minmax("ct.minimum", true);
  case BinaryOperator::LogicalAnd:
  case BinaryOperator::BitwiseAnd: return infix("&");
  case BinaryOperator::LogicalOr:
  case BinaryOperator::BitwiseOr: return infix("|");
  case BinaryOperator::BitwiseXor: return infix("^");
  case BinaryOperator::LeftShift: return infix("<<");
  case BinaryOperator::RightShift: return infix(">>");
  }
  return operation.emitOpError("has no cuTile binary source translation"),
         failure();
}

FailureOr<Translation> translate(gpu::UnaryOp operation,
                                 ArrayRef<std::string> operands) {
  auto expression = [&](std::string value) -> FailureOr<Translation> {
    return Translation{std::move(value), {}};
  };
  auto call = [&](StringRef function) {
    return expression(function.str() + "(" + operands[0] + ")");
  };
  auto library = [&](StringRef function) -> FailureOr<Translation> {
    if (Emitter::elementType(operation.getInput().getType()).isF64())
      return operation.emitOpError(
          "cuTile library lowering currently requires f32 or narrower inputs"),
             failure();
    return Translation{"cutile_math." + function.str() + "(" + operands[0] +
                       ")", {libraryMathImport.str()}};
  };
  switch (operation.getOperatorKind()) {
  case UnaryOperator::Negate: return expression("(-" + operands[0] + ")");
  case UnaryOperator::Not: return expression("(~" + operands[0] + ")");
  case UnaryOperator::Exp: return call("ct.exp");
  case UnaryOperator::Exp2:
    if (operation.getApproximate())
      return expression("ct.exp2(" + operands[0] + ", flush_to_zero=" +
                         (operation.getFlushToZero() ? "True" : "False") + ")");
    return call("ct.exp2");
  case UnaryOperator::Log: return call("ct.log");
  case UnaryOperator::Sin: return call("ct.sin");
  case UnaryOperator::Cos: return call("ct.cos");
  case UnaryOperator::Floor: return call("ct.floor");
  case UnaryOperator::Rsqrt: return call("ct.rsqrt");
  case UnaryOperator::Sigmoid: {
    std::string computation = Emitter::elementType(
        operation.getInput().getType()).isF64() ? "ct.float64" : "ct.float32";
    return expression("ct.astype(1.0 / (1.0 + ct.exp(-ct.astype(" + operands[0] +
                      ", " + computation + "))), " +
                      typeName(operation.getResult().getType()) + ")");
  }
  case UnaryOperator::Tanh:
    if (operation.getApproximate()) {
      auto function = operation->getParentOfType<func::FuncOp>();
      auto capabilities = function
          ? function->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr)
          : gpu::CapabilitiesAttr();
      if (!capabilities ||
          10 * capabilities.getComputeCapabilityMajor() +
              capabilities.getComputeCapabilityMinor() < 75)
        return operation.emitOpError(
            "native approximate tanh requires compute capability 7.5 or newer"),
               failure();
      return expression("ct.tanh(" + operands[0] +
                        ", rounding_mode=ct.RoundingMode.APPROX)");
    }
    return call("ct.tanh");
  case UnaryOperator::Abs: return call("ct.abs");
  case UnaryOperator::Sqrt: return call("ct.sqrt");
  case UnaryOperator::Erf: return library("erf");
  case UnaryOperator::Log1p: return library("log1p");
  case UnaryOperator::Lgamma: return library("lgamma");
  case UnaryOperator::Erfc: return library("erfc");
  case UnaryOperator::I0: return library("i0");
  case UnaryOperator::Asin: return library("asin");
  }
  return operation.emitOpError("has no cuTile unary source translation"), failure();
}

FailureOr<Translation> translate(gpu::CompareOp operation,
                                 ArrayRef<std::string> operands) {
  StringRef token;
  switch (operation.getPredicate()) {
  case ComparePredicate::Eq: token = "=="; break;
  case ComparePredicate::Ne: token = "!="; break;
  case ComparePredicate::Lt: token = "<"; break;
  case ComparePredicate::Le: token = "<="; break;
  case ComparePredicate::Gt: token = ">"; break;
  case ComparePredicate::Ge: token = ">="; break;
  }
  return Translation{"(" + operands[0] + " " + token.str() + " " +
                     operands[1] + ")", {}, true};
}

FailureOr<Translation> translate(gpu::CastOp operation,
                                 ArrayRef<std::string> operands) {
  return Translation{castExpression(operands[0], operation.getValue().getType(),
                                    operation.getResult().getType(), false),
                     {}};
}

FailureOr<Translation> translate(gpu::BitcastOp operation,
                                 ArrayRef<std::string> operands) {
  return Translation{castExpression(operands[0], operation.getValue().getType(),
                                    operation.getResult().getType(), true),
                     {}};
}

FailureOr<Translation> translate(gpu::SelectOp,
                                 ArrayRef<std::string> operands) {
  return Translation{selectExpression(operands), {}};
}

template <typename Op> FailureOr<Translation> query(Op operation) {
  SmallVector<std::string> operands(operation->getNumOperands(), "value");
  return translate(operation, operands);
}

template <typename Op> void addNumericalOperation(Emitter::Emitters &emitters) {
  emitters.add<Op>(
      [](Op operation) { return failure(failed(query(operation))); },
      [](Op operation, Emitter &emitter) -> LogicalResult {
        SmallVector<std::string> operands;
        for (Value value : operation->getOperands())
          operands.push_back(emitter.valueString(value));
        auto translation = translate(operation, operands);
        if (failed(translation))
          return failure();
        Value result = operation->getResult(0);
        bool compileTime = translation->propagatesConstexpr &&
            (result.getType().isIndex() || result.getType().isInteger(1)) &&
            llvm::all_of(operation->getOperands(), [&](Value value) {
              return emitter.isConstexprValue(value);
            });
        emitter.assign(result, translation->expression, compileTime);
        return success();
      },
      [](Op operation) -> FailureOr<Dependencies> {
        auto translation = query(operation);
        if (failed(translation))
          return failure();
        return std::move(translation->dependencies);
      });
}

} // namespace

std::string integerDivision(gpu::PhysicalExprKind kind, StringRef lhs,
                            StringRef rhs) {
  switch (kind) {
  case gpu::PhysicalExprKind::FloorDiv:
    return "ct.floordiv(" + lhs.str() + ", " + rhs.str() + ")";
  case gpu::PhysicalExprKind::CeilDiv:
    return "ct.cdiv(" + lhs.str() + ", " + rhs.str() + ")";
  default: llvm_unreachable("expected a physical integer division");
  }
}

void addNumericalOperations(Emitter::Emitters &emitters) {
  addNumericalOperation<gpu::BinaryOp>(emitters);
  addNumericalOperation<gpu::UnaryOp>(emitters);
  addNumericalOperation<gpu::CompareOp>(emitters);
  addNumericalOperation<gpu::CastOp>(emitters);
  addNumericalOperation<gpu::BitcastOp>(emitters);
  addNumericalOperation<gpu::SelectOp>(emitters);
}

std::string numericalCast(Emitter &emitter, Value value, Type result,
                          bool bitcast) {
  return castExpression(emitter.valueString(value), value.getType(), result,
                        bitcast);
}

std::string numericalSelect(Emitter &emitter, gpu::SelectOp operation) {
  SmallVector<std::string> operands;
  for (Value value : operation->getOperands())
    operands.push_back(emitter.valueString(value));
  return selectExpression(operands);
}

} // namespace intent::cutile
