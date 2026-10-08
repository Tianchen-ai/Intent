#include "Intent/Serialization/Scalar.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace intent {

bool isStandardScalarOperation(Operation *operation) {
  StringRef dialect = operation->getName().getDialectNamespace();
  return dialect == "arith" || dialect == "math";
}

unsigned scalarIntegerWidth(Type type) {
  type = getElementTypeOrSelf(type);
  return type.isIndex() ? 64 : cast<IntegerType>(type).getWidth();
}

StringRef scalarComparisonToken(ScalarRelation relation) {
  switch (relation) {
  case ScalarRelation::Equal: return "==";
  case ScalarRelation::NotEqual: return "!=";
  case ScalarRelation::Less: return "<";
  case ScalarRelation::LessEqual: return "<=";
  case ScalarRelation::Greater: return ">";
  case ScalarRelation::GreaterEqual: return ">=";
  }
  llvm_unreachable("invalid scalar relation");
}

StringRef scalarComparisonMethod(ScalarRelation relation) {
  switch (relation) {
  case ScalarRelation::Equal: return "eq";
  case ScalarRelation::NotEqual: return "ne";
  case ScalarRelation::Less: return "lt";
  case ScalarRelation::LessEqual: return "le";
  case ScalarRelation::Greater: return "gt";
  case ScalarRelation::GreaterEqual: return "ge";
  }
  llvm_unreachable("invalid scalar relation");
}

FailureOr<ScalarOperation> ScalarOperation::read(Operation *operation) {
  if (operation->getNumResults() != 1)
    return operation->emitError("scalar source emission requires one result"), failure();
  auto decoded = llvm::TypeSwitch<Operation *, std::optional<ScalarOperation>>(operation)
#define INTENT_SCALAR_OP(Op, Kind, Cast)                                       \
      .Case<Op>([&](auto) {                                                   \
        return ScalarOperation{operation, ScalarKind::Kind, {},               \
            ScalarCast::Cast == ScalarCast::None ? std::nullopt               \
                : std::optional<ScalarCast>(ScalarCast::Cast), {}};           \
      })
#include "Intent/Serialization/ScalarOps.def"
#undef INTENT_SCALAR_OP
      .Default([](Operation *) -> std::optional<ScalarOperation> { return std::nullopt; });
  if (!decoded)
    return operation->emitError("standard scalar operation has no source emission semantics"), failure();
  ScalarOperation result = *decoded;
  if (auto constant = dyn_cast<arith::ConstantOp>(operation))
    result.constant = constant.getValue();
  if (auto trunc = dyn_cast<arith::TruncFOp>(operation))
    if (auto rounding = trunc.getRoundingmode();
        rounding && *rounding != arith::RoundingMode::to_nearest_even)
      return operation->emitError("source emission requires the declared floating rounding mode to be implemented"), failure();
  using R = ScalarRelation;
  if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
    switch (compare.getPredicate()) {
    case arith::CmpIPredicate::eq: result.comparison.relation = R::Equal; break;
    case arith::CmpIPredicate::ne: result.comparison.relation = R::NotEqual; break;
    case arith::CmpIPredicate::slt: result.comparison.relation = R::Less; break;
    case arith::CmpIPredicate::sle: result.comparison.relation = R::LessEqual; break;
    case arith::CmpIPredicate::sgt: result.comparison.relation = R::Greater; break;
    case arith::CmpIPredicate::sge: result.comparison.relation = R::GreaterEqual; break;
    case arith::CmpIPredicate::ult: result.comparison = {R::Less, ScalarNaN::None, true}; break;
    case arith::CmpIPredicate::ule: result.comparison = {R::LessEqual, ScalarNaN::None, true}; break;
    case arith::CmpIPredicate::ugt: result.comparison = {R::Greater, ScalarNaN::None, true}; break;
    case arith::CmpIPredicate::uge: result.comparison = {R::GreaterEqual, ScalarNaN::None, true}; break;
    }
  }
  if (auto compare = dyn_cast<arith::CmpFOp>(operation)) {
    auto &comparison = result.comparison;
    comparison.nan = ScalarNaN::Ordered;
    switch (compare.getPredicate()) {
    case arith::CmpFPredicate::AlwaysFalse: comparison.nan = ScalarNaN::AlwaysFalse; break;
    case arith::CmpFPredicate::AlwaysTrue: comparison.nan = ScalarNaN::AlwaysTrue; break;
    case arith::CmpFPredicate::OEQ: comparison.relation = R::Equal; break;
    case arith::CmpFPredicate::ONE: comparison.relation = R::NotEqual; break;
    case arith::CmpFPredicate::OLT: comparison.relation = R::Less; break;
    case arith::CmpFPredicate::OLE: comparison.relation = R::LessEqual; break;
    case arith::CmpFPredicate::OGT: comparison.relation = R::Greater; break;
    case arith::CmpFPredicate::OGE: comparison.relation = R::GreaterEqual; break;
    case arith::CmpFPredicate::ORD: comparison.relationResult = false; break;
    case arith::CmpFPredicate::UEQ: comparison = {R::Equal, ScalarNaN::Unordered}; break;
    case arith::CmpFPredicate::UNE: comparison = {R::NotEqual, ScalarNaN::Unordered}; break;
    case arith::CmpFPredicate::ULT: comparison = {R::Less, ScalarNaN::Unordered}; break;
    case arith::CmpFPredicate::ULE: comparison = {R::LessEqual, ScalarNaN::Unordered}; break;
    case arith::CmpFPredicate::UGT: comparison = {R::Greater, ScalarNaN::Unordered}; break;
    case arith::CmpFPredicate::UGE: comparison = {R::GreaterEqual, ScalarNaN::Unordered}; break;
    case arith::CmpFPredicate::UNO: comparison = {R::Equal, ScalarNaN::Unordered, false, false}; break;
    }
  }
  return result;
}

LogicalResult verifyScalarEmission(Operation *operation, ScalarRenderer renderer) {
  SmallVector<std::string> operands(operation->getNumOperands(), "value");
  return failure(failed(renderer(operation, operands)));
}

namespace {
std::string parenthesize(StringRef expression) { return "(" + expression.str() + ")"; }
std::string cCast(StringRef type, StringRef value) {
  return "((" + type.str() + ")(" + value.str() + "))";
}
std::string unsignedType(Type type) {
  return "uint" + std::to_string(std::max(8u, scalarIntegerWidth(type))) + "_t";
}
std::string signedValue(Type type, StringRef value, CScalarTypes types) {
  // i1 is a one-bit integer. Signed extension/comparison interprets true as -1.
  return type.isInteger(1) ? "(-((int32_t)(" + value.str() + ")))"
                           : cCast(types(type)->name, value);
}
} // namespace

FailureOr<std::string> emitCScalar(const ScalarOperation &scalar,
                                  ArrayRef<std::string> operands,
                                  CScalarTypes types) {
  Operation *operation = scalar.operation;
  auto reject = [&](StringRef message) -> FailureOr<std::string> {
    operation->emitError(message);
    return failure();
  };
  for (Type type : llvm::concat<Type>(operation->getOperandTypes(), operation->getResultTypes()))
    if (isa<ShapedType>(type) || !types(type))
      return reject("scalar type has no C source representation");
  Type type = scalar.type();
  CScalarType resultType = *types(type);
  auto convertResult = [&](std::string value) {
    if (!resultType.encode.empty()) return resultType.encode + "(" + value + ")";
    if (type.isInteger(1)) value = "((uint8_t)(" + value + ") & 1)";
    return cCast(resultType.name, value);
  };
  auto input = [&](unsigned index, bool unsignedInteger = false) {
    Type inputType = scalar.inputType(index);
    auto spelling = *types(inputType);
    if (unsignedInteger) return cCast(unsignedType(inputType), operands[index]);
    if (!spelling.decode.empty()) return spelling.decode + "(" + operands[index] + ")";
    return parenthesize(operands[index]);
  };
  auto signedInput = [&](unsigned index) {
    return signedValue(scalar.inputType(index), operands[index], types);
  };
  using K = ScalarKind;
  if (scalar.kind == K::Constant) {
    if (auto integer = dyn_cast<IntegerAttr>(scalar.constant)) {
      llvm::SmallString<32> bits;
      integer.getValue().toString(bits, 10, false);
      return convertResult(bits.str().str() + "ULL");
    }
    auto floating = dyn_cast<FloatAttr>(scalar.constant);
    if (!floating) return reject("C scalar constant requires an integer or floating attribute");
    auto value = floating.getValue();
    std::string text;
    if (value.isNaN()) text = "NAN";
    else if (value.isInfinity()) text = value.isNegative() ? "(-INFINITY)" : "INFINITY";
    else {
      llvm::SmallString<32> literal;
      value.toString(literal);
      text = literal.str().str();
      if (text.find_first_of(".eE") == std::string::npos) text += ".0";
      if (!type.isF64()) text += "f";
    }
    return convertResult(text);
  }
  if (scalar.kind == K::Cast) {
    auto kind = *scalar.cast;
    std::string value = input(0);
    if (kind == ScalarCast::ExtendUnsigned || kind == ScalarCast::UnsignedToFloat ||
        kind == ScalarCast::IndexUnsigned)
      value = input(0, true);
    if (kind == ScalarCast::ExtendSigned || kind == ScalarCast::SignedToFloat ||
        kind == ScalarCast::IndexSigned)
      value = signedInput(0);
    if (kind == ScalarCast::FloatToUnsigned) value = cCast(unsignedType(type), value);
    if (kind == ScalarCast::FloatToSigned && type.isInteger(1))
      value = cCast("int8_t", value);
    return convertResult(value);
  }
  if (scalar.kind == K::Bitcast)
    return reject("this C scalar surface has no bitcast expression");
  if (scalar.kind == K::Select)
    return "(" + operands[0] + " ? " + operands[1] + " : " + operands[2] + ")";
  if (scalar.kind == K::Compare) {
    const auto &comparison = scalar.comparison;
    if (comparison.nan == ScalarNaN::AlwaysFalse) return std::string("0");
    if (comparison.nan == ScalarNaN::AlwaysTrue) return std::string("1");
    bool floating = isa<FloatType>(scalar.inputType());
    std::string lhs = floating || comparison.unsignedInteger
        ? input(0, comparison.unsignedInteger) : signedInput(0);
    std::string rhs = floating || comparison.unsignedInteger
        ? input(1, comparison.unsignedInteger) : signedInput(1);
    std::string relation = "(" + lhs + " " + scalarComparisonToken(comparison.relation).str() + " " + rhs + ")";
    if (!floating) return relation;
    std::string unordered = "(isnan(" + lhs + ") || isnan(" + rhs + "))";
    if (!comparison.relationResult)
      return comparison.nan == ScalarNaN::Ordered ? "(!" + unordered + ")" : unordered;
    return comparison.nan == ScalarNaN::Ordered
        ? "(!" + unordered + " && " + relation + ")"
        : "(" + unordered + " || " + relation + ")";
  }
  std::string lhs = input(0), rhs = operands.size() > 1 ? input(1) : "";
  auto binary = [&](StringRef token) { return "(" + lhs + " " + token.str() + " " + rhs + ")"; };
  auto function = [&](StringRef name) {
    std::string callee = name.str() + (type.isF64() ? "" : "f");
    std::string arguments;
    for (unsigned i = 0; i < operands.size(); ++i) {
      if (i) arguments += ", ";
      arguments += input(i);
    }
    return callee + "(" + arguments + ")";
  };
  std::string expression;
  switch (scalar.kind) {
  case K::Add: case K::Subtract: case K::Multiply: case K::ShiftLeft: {
    StringRef token = scalar.kind == K::Add ? "+" : scalar.kind == K::Subtract ? "-" : scalar.kind == K::Multiply ? "*" : "<<";
    if (type.isIntOrIndex()) {
      // Avoid both signed overflow and the integer promotion of narrow lanes.
      StringRef carrier = scalarIntegerWidth(type) == 64 ? "uint64_t" : "uint32_t";
      lhs = cCast(carrier, input(0, true));
      rhs = cCast(carrier, input(1, true));
    }
    expression = binary(token);
    break;
  }
  case K::DivideSigned: case K::RemainderSigned:
  case K::FloorDivideSigned: case K::CeilDivideSigned: {
    lhs = signedInput(0); rhs = signedInput(1);
    expression = binary(scalar.kind == K::RemainderSigned ? "%" : "/");
    // Unlike divsi, remsi defines MIN % -1. C's remainder shares the division
    // overflow trap, so avoid evaluating that case in the generated program.
    if (scalar.kind == K::RemainderSigned)
      expression = "(" + rhs + " == -1 ? 0 : " + expression + ")";
    if (scalar.kind == K::FloorDivideSigned || scalar.kind == K::CeilDivideSigned) {
      bool floor = scalar.kind == K::FloorDivideSigned;
      expression = "(" + expression + (floor ? " - " : " + ") + "(" + binary("%") +
          " != 0 && ((" + lhs + " < 0) " + (floor ? "!=" : "==") + " (" + rhs + " < 0))))";
    }
    break;
  }
  case K::DivideUnsigned: case K::RemainderUnsigned: case K::ShiftRightUnsigned:
  case K::CeilDivideUnsigned:
    lhs = input(0, true); rhs = input(1, true);
    expression = binary(scalar.kind == K::RemainderUnsigned ? "%" : scalar.kind == K::ShiftRightUnsigned ? ">>" : "/");
    if (scalar.kind == K::CeilDivideUnsigned) expression = "(" + expression + " + (" + binary("%") + " != 0))";
    break;
  case K::ShiftRightSigned: lhs = signedInput(0); expression = binary(">>"); break;
  case K::DivideFloat: expression = binary("/"); break;
  case K::And: expression = binary("&"); break;
  case K::Or: expression = binary("|"); break;
  case K::Xor: expression = binary("^"); break;
  case K::MinimumSigned: case K::MaximumSigned: case K::MinimumUnsigned: case K::MaximumUnsigned: {
    bool isUnsigned = scalar.kind == K::MinimumUnsigned || scalar.kind == K::MaximumUnsigned;
    lhs = isUnsigned ? input(0, true) : signedInput(0);
    rhs = isUnsigned ? input(1, true) : signedInput(1);
    bool maximum = scalar.kind == K::MaximumSigned || scalar.kind == K::MaximumUnsigned;
    expression = "(" + binary(maximum ? ">" : "<") + " ? " + lhs + " : " + rhs + ")";
    break;
  }
  case K::MinimumFloat: case K::MaximumFloat: case K::MinimumNumber: case K::MaximumNumber: {
    bool maximum = scalar.kind == K::MaximumFloat || scalar.kind == K::MaximumNumber;
    expression = function(maximum ? "fmax" : "fmin");
    // IEEE min/max distinguish signed zeros. Do not delegate a zero tie to a
    // library whose tie choice can depend on operand ordering.
    std::string zero = "(" + lhs + " == 0 && " + rhs + " == 0)";
    std::string negative = "(signbit(" + lhs + ") " + (maximum ? "&&" : "||") + " signbit(" + rhs + "))";
    expression = "(" + zero + " ? (" + negative + " ? -0.0 : 0.0) : " + expression + ")";
    if (scalar.kind == K::MinimumFloat || scalar.kind == K::MaximumFloat)
      expression = "((isnan(" + lhs + ") || isnan(" + rhs + ")) ? NAN : " + expression + ")";
    break;
  }
  case K::Negate: expression = "(-" + lhs + ")"; break;
  case K::AbsInteger:
    lhs = signedInput(0);
    expression = "(" + lhs + " < 0 ? (uint64_t)(0) - (uint64_t)(" + input(0, true) + ") : (uint64_t)(" + input(0, true) + "))";
    break;
  case K::AbsFloat: expression = function("fabs"); break;
  case K::Exp: expression = function("exp"); break;
  case K::Exp2: expression = function("exp2"); break;
  case K::Log: expression = function("log"); break;
  case K::Sqrt: expression = function("sqrt"); break;
  case K::Rsqrt:
    expression = std::string(type.isF64() ? "(1.0 / " : "(1.0f / ") + function("sqrt") + ")";
    break;
  case K::Tanh: expression = function("tanh"); break;
  case K::Sin: expression = function("sin"); break;
  case K::Cos: expression = function("cos"); break;
  case K::Floor: expression = function("floor"); break;
  case K::Erf: expression = function("erf"); break;
  case K::Power: expression = function("pow"); break;
  case K::Fma: expression = function("fma"); break;
  case K::Constant: case K::Cast: case K::Bitcast: case K::Select: case K::Compare:
    llvm_unreachable("handled before scalar expression switch");
  }
  return convertResult(expression);
}

} // namespace intent
