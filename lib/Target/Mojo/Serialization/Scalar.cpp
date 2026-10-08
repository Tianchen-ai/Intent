#include "Intent/Target/Mojo/Serialization/Scalar.h"
#include "Intent/Serialization/ScalarEmitters.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
namespace intent::mojo {

bool supportsScalarType(Type type) {
  if (auto vector = dyn_cast<VectorType>(type)) {
    int64_t width = vector.getNumElements();
    return !vector.isScalable() && vector.getRank() == 1 && width > 0 &&
        (width & (width - 1)) == 0 && supportsScalarType(vector.getElementType());
  }
  return type.isIndex() || type.isF16() || type.isBF16() || type.isF32() ||
      type.isF64() || isa<Float8E4M3FNType, Float8E5M2Type>(type) ||
      type.isSignlessInteger(1) || type.isSignlessInteger(8) ||
      type.isSignlessInteger(16) || type.isSignlessInteger(32) ||
      type.isSignlessInteger(64);
}

std::string scalarDType(Type type) {
  type = getElementTypeOrSelf(type);
  if (type.isF16()) return "float16";
  if (type.isBF16()) return "bfloat16";
  if (type.isF32()) return "float32";
  if (type.isF64()) return "float64";
  if (isa<Float8E4M3FNType>(type)) return "float8_e4m3fn";
  if (isa<Float8E5M2Type>(type)) return "float8_e5m2";
  if (type.isInteger(1)) return "bool";
  if (type.isIndex()) return "int64";
  return "int" + std::to_string(cast<IntegerType>(type).getWidth());
}

std::string scalarValueType(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    return "SIMD[DType." + scalarDType(type) + ", " +
        std::to_string(vector.getNumElements()) + "]";
  if (type.isIndex()) return "Int";
  if (type.isInteger(1)) return "Bool";
  return "SIMD[DType." + scalarDType(type) + ", 1]";
}

namespace {
std::string floatingLiteral(Type type, const llvm::APFloat &value) {
  std::string spelling = scalarValueType(type);
  if (!value.isFinite())
    return "(" + spelling + "(" + (value.isNaN() ? "0" : value.isNegative() ? "-1" : "1") +
        ") / " + spelling + "(0))";
  if (value.isZero() && value.isNegative()) return spelling + "(-0.0)";
  llvm::SmallString<32> literal;
  value.toString(literal);
  return spelling + "(" + literal.str().str() + ")";
}

FailureOr<std::string> render(Operation *operation, ArrayRef<std::string> operands) {
  auto decoded = ScalarOperation::read(operation);
  if (failed(decoded)) return failure();
  const auto &scalar = *decoded;
  for (Type type : llvm::concat<Type>(operation->getOperandTypes(), operation->getResultTypes()))
    if (!supportsScalarType(type))
      return operation->emitError("scalar type has no Mojo source representation: ") << type, failure();
  Type type = scalar.type();
  auto parenthesize = [](StringRef value) { return "(" + value.str() + ")"; };
  auto simdInput = [&](unsigned index) {
    Type inputType = scalar.inputType(index);
    return inputType.isIndex() ? "Int64(" + operands[index] + ")"
        : inputType.isInteger(1) ? "SIMD[DType.bool, 1](" + operands[index] + ")"
                               : parenthesize(operands[index]);
  };
  auto unsignedInput = [&](unsigned index) {
    auto inputType = getElementTypeOrSelf(scalar.inputType(index));
    if (inputType.isInteger(1)) return simdInput(index) + ".cast[DType.uint8]()";
    return "bitcast[DType.uint" + std::to_string(scalarIntegerWidth(inputType)) + "](" + simdInput(index) + ")";
  };
  auto signedInput = [&](unsigned index) {
    if (getElementTypeOrSelf(scalar.inputType(index)).isInteger(1))
      return "(-" + simdInput(index) + ".cast[DType.int8]())";
    return simdInput(index);
  };
  auto fromUnsigned = [&](std::string value) {
    if (getElementTypeOrSelf(type).isInteger(1)) {
      std::string lowBit = "((" + value + ") & 1)";
      return isa<VectorType>(type) ? lowBit + ".cast[DType.bool]()" : "Bool(" + lowBit + ")";
    }
    std::string converted = "bitcast[DType." + scalarDType(type) + "](" + value + ")";
    return type.isIndex() ? "Int(" + converted + ")" : converted;
  };
  auto compare = [&](StringRef lhs, StringRef rhs, ScalarRelation relation,
                     bool vector) {
    return vector ? parenthesize(lhs) + "." + scalarComparisonMethod(relation).str() + "(" + rhs.str() + ")"
        : "(" + lhs.str() + " " + scalarComparisonToken(relation).str() + " " + rhs.str() + ")";
  };
  using K = ScalarKind;
  if (scalar.kind == K::Constant) {
    if (auto integer = dyn_cast<IntegerAttr>(scalar.constant)) {
      if (type.isInteger(1)) return std::string(integer.getValue().isZero() ? "False" : "True");
      llvm::SmallString<32> literal;
      integer.getValue().toString(literal, 10, true);
      return scalarValueType(type) + "(" + literal.str().str() + ")";
    }
    if (auto floating = dyn_cast<FloatAttr>(scalar.constant))
      return floatingLiteral(type, floating.getValue());
    auto dense = dyn_cast<DenseElementsAttr>(scalar.constant);
    if (!dense)
      return operation->emitError("Mojo constant requires numeric scalar or dense vector values"), failure();
    SmallVector<std::string> elements;
    if (isa<FloatType>(dense.getElementType())) {
      for (llvm::APFloat value : dense.getValues<llvm::APFloat>()) {
        elements.push_back(floatingLiteral(dense.getElementType(), value));
        if (dense.isSplat()) break;
      }
    } else {
      for (llvm::APInt value : dense.getValues<llvm::APInt>()) {
        llvm::SmallString<32> literal;
        value.toString(literal, 10, true);
        elements.push_back(dense.getElementType().isInteger(1)
            ? (value.isZero() ? "False" : "True") : literal.str().str());
        if (dense.isSplat()) break;
      }
    }
    std::string fill = dense.isSplat() && dense.getElementType().isInteger(1) ? "fill=" : "";
    return scalarValueType(type) + "(" + fill + llvm::join(elements, ", ") + ")";
  }
  if (scalar.kind == K::Compare) {
    auto &predicate = scalar.comparison;
    bool vector = isa<VectorType>(scalar.inputType());
    if (predicate.nan == ScalarNaN::AlwaysFalse || predicate.nan == ScalarNaN::AlwaysTrue) {
      std::string value = predicate.nan == ScalarNaN::AlwaysTrue ? "True" : "False";
      return vector ? scalarValueType(type) + "(fill=" + value + ")" : value;
    }
    bool floating = isa<FloatType>(getElementTypeOrSelf(scalar.inputType()));
    auto input = [&](unsigned index) {
      return floating ? operands[index] : predicate.unsignedInteger ? unsignedInput(index) : signedInput(index);
    };
    std::string lhs = input(0), rhs = input(1);
    std::string relation = compare(lhs, rhs, predicate.relation, vector);
    if (!floating) return relation;
    bool ordered = predicate.nan == ScalarNaN::Ordered;
    auto selfRelation = ordered ? ScalarRelation::Equal : ScalarRelation::NotEqual;
    std::string validity = "(" + compare(lhs, lhs, selfRelation, vector) +
        (ordered ? " & " : " | ") + compare(rhs, rhs, selfRelation, vector) + ")";
    if (!predicate.relationResult) return validity;
    // The common comparisons have the correct NaN behavior natively.
    if ((ordered && predicate.relation != ScalarRelation::NotEqual) ||
        (!ordered && predicate.relation == ScalarRelation::NotEqual))
      return relation;
    return "(" + validity + (ordered ? " & " : " | ") + relation + ")";
  }
  if (scalar.kind == K::Cast) {
    auto conversion = *scalar.cast;
    std::string value = simdInput(0);
    if (conversion == ScalarCast::ExtendSigned || conversion == ScalarCast::SignedToFloat ||
        conversion == ScalarCast::IndexSigned) value = signedInput(0);
    if (conversion == ScalarCast::ExtendUnsigned || conversion == ScalarCast::UnsignedToFloat ||
        conversion == ScalarCast::IndexUnsigned) value = unsignedInput(0);
    if (getElementTypeOrSelf(type).isInteger(1)) {
      // trunc(iN -> i1) keeps bit zero; a truth conversion would keep any bit.
      if (conversion == ScalarCast::FloatToSigned || conversion == ScalarCast::FloatToUnsigned)
        value += ".cast[DType.int8]()";
      std::string lowBit = "(" + value + " & 1)";
      return isa<VectorType>(type) ? lowBit + ".ne(0)" : "Bool(" + lowBit + ")";
    }
    if (conversion == ScalarCast::FloatToUnsigned) {
      value += ".cast[DType.uint" + std::to_string(scalarIntegerWidth(type)) + "]()";
      return fromUnsigned(value);
    }
    value += ".cast[DType." + scalarDType(type) + "]()";
    return type.isIndex() ? "Int(" + value + ")" : value;
  }
  if (scalar.kind == K::Bitcast) {
    std::string expression = "bitcast[DType." + scalarDType(type) + "](" + simdInput(0) + ")";
    if (type.isInteger(1)) return "Bool(" + expression + ")";
    return type.isIndex() ? "Int(" + expression + ")" : expression;
  }
  if (scalar.kind == K::Select)
    return isa<VectorType>(scalar.inputType())
        ? "(" + operands[0] + ").select(" + operands[1] + ", " + operands[2] + ")"
        : "(" + operands[1] + " if " + operands[0] + " else " + operands[2] + ")";
  std::string lhs = parenthesize(operands[0]);
  std::string rhs = operands.size() > 1 ? parenthesize(operands[1]) : "";
  auto binary = [&](StringRef token) { return "(" + lhs + " " + token.str() + " " + rhs + ")"; };
  auto call = [&](StringRef function) { return function.str() + "(" + llvm::join(operands, ", ") + ")"; };
  switch (scalar.kind) {
  case K::Add: return binary(getElementTypeOrSelf(type).isInteger(1) ? "^" : "+");
  case K::Subtract: return binary(getElementTypeOrSelf(type).isInteger(1) ? "^" : "-");
  case K::Multiply: return binary(getElementTypeOrSelf(type).isInteger(1) ? "&" : "*");
  case K::ShiftLeft:
    if (getElementTypeOrSelf(type).isInteger(1)) {
      lhs = unsignedInput(0); rhs = unsignedInput(1);
      return fromUnsigned(binary("<<"));
    }
    return binary("<<");
  case K::ShiftRightSigned:
    if (getElementTypeOrSelf(type).isInteger(1)) return operands[0];
    return binary(">>");
  case K::And: return binary("&");
  case K::Or: return binary("|");
  case K::Xor: return binary("^");
  case K::DivideFloat: return binary("/");
  case K::Power: return binary("**");
  case K::DivideSigned: case K::RemainderSigned: {
    if (scalar.kind == K::RemainderSigned && getElementTypeOrSelf(type).isInteger(1))
      return isa<VectorType>(type) ? scalarValueType(type) + "(fill=False)" : std::string("False");
    lhs = signedInput(0); rhs = signedInput(1);
    if (scalar.kind == K::RemainderSigned) {
      // srem has no MIN / -1 overflow. Replacing that divisor by one keeps
      // remainder zero and lets the existing truncating-division spelling run.
      rhs = isa<VectorType>(type)
          ? "(" + rhs + ").eq(-1).select(1, " + rhs + ")"
          : "(" + scalarValueType(type.isIndex() ? IntegerType::get(type.getContext(), 64) : type) +
              "(1) if " + rhs + " == -1 else " + rhs + ")";
    }
    std::string quotient = binary("/");
    std::string expression = scalar.kind == K::RemainderSigned
        ? "(" + lhs + " - (" + quotient + ") * " + rhs + ")" : quotient;
    if (getElementTypeOrSelf(type).isInteger(1)) {
      expression = "((" + expression + ") & 1)";
      return isa<VectorType>(type) ? expression + ".cast[DType.bool]()" : "Bool(" + expression + ")";
    }
    return type.isIndex() ? "Int(" + expression + ")" : expression;
  }
  case K::DivideUnsigned: case K::RemainderUnsigned: case K::ShiftRightUnsigned:
  case K::MinimumUnsigned: case K::MaximumUnsigned:
    lhs = unsignedInput(0); rhs = unsignedInput(1);
    return fromUnsigned(scalar.kind == K::MinimumUnsigned ? "min(" + lhs + ", " + rhs + ")"
        : scalar.kind == K::MaximumUnsigned ? "max(" + lhs + ", " + rhs + ")"
        : binary(scalar.kind == K::DivideUnsigned ? "/" : scalar.kind == K::RemainderUnsigned ? "%" : ">>"));
  case K::FloorDivideSigned: case K::CeilDivideSigned: case K::CeilDivideUnsigned:
    return operation->emitError("Mojo integer ceil/floor division must be expanded during legalization"), failure();
  case K::MinimumSigned: case K::MaximumSigned:
    if (getElementTypeOrSelf(type).isInteger(1))
      return binary(scalar.kind == K::MinimumSigned ? "|" : "&");
    return call(scalar.kind == K::MinimumSigned ? "min" : "max");
  case K::MinimumFloat: case K::MaximumFloat: case K::MinimumNumber: case K::MaximumNumber: {
    StringRef intrinsic = scalar.kind == K::MinimumFloat ? "llvm.minimum"
        : scalar.kind == K::MaximumFloat ? "llvm.maximum"
        : scalar.kind == K::MinimumNumber ? "llvm.minimumnum" : "llvm.maximumnum";
    return "llvm_intrinsic[\"" + intrinsic.str() + "\", " + scalarValueType(type) + "](" + llvm::join(operands, ", ") + ")";
  }
  case K::Negate: return "(-" + lhs + ")";
  case K::AbsInteger:
    if (getElementTypeOrSelf(type).isInteger(1)) return operands[0];
    return call("abs");
  case K::AbsFloat: return call("abs");
  case K::Exp: return call("exp");
  case K::Exp2: return call("exp2");
  case K::Log: return call("log");
  case K::Sqrt: return call("sqrt");
  case K::Rsqrt: return operation->emitError("Mojo rsqrt must be expanded during legalization"), failure();
  case K::Tanh: return call("tanh");
  case K::Sin: return call("sin");
  case K::Cos: return call("cos");
  case K::Floor: return call("floor");
  case K::Erf: return call("erf");
  case K::Fma: return call("fma");
  case K::Constant: case K::Compare: case K::Cast: case K::Bitcast: case K::Select:
    llvm_unreachable("handled before scalar expression switch");
  }
  llvm_unreachable("invalid scalar operation kind");
}

const auto &emitters() {
  static const auto table = scalarExpressionEmitters(render);
  return table;
}
} // namespace

FailureOr<std::string> emitScalar(Operation *operation, ArrayRef<std::string> operands) {
  return emitScalarExpression(emitters(), operation, operands);
}

LogicalResult verifyScalar(Operation *operation) {
  return emitters().verify(operation);
}
} // namespace intent::mojo
