#ifndef INTENT_SERIALIZATION_SCALAR_H
#define INTENT_SERIALIZATION_SCALAR_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>
#include <string>

namespace intent {

// The meaning of a standard scalar (or elementwise vector) operation. This is
// read from current IR; it is not an alternative arithmetic IR or lowering plan.
enum class ScalarKind {
  Constant, Add, Subtract, Multiply, DivideSigned, DivideUnsigned, DivideFloat,
  RemainderSigned, RemainderUnsigned, FloorDivideSigned, CeilDivideSigned,
  CeilDivideUnsigned, And, Or, Xor, ShiftLeft, ShiftRightSigned,
  ShiftRightUnsigned, MinimumSigned, MaximumSigned, MinimumUnsigned,
  MaximumUnsigned, MinimumFloat, MaximumFloat, MinimumNumber, MaximumNumber,
  Compare, Select, Cast, Bitcast, Negate, AbsInteger, AbsFloat, Exp, Exp2, Log,
  Sqrt, Rsqrt, Tanh, Sin, Cos, Floor, Erf, Power, Fma
};

enum class ScalarCast {
  None, ExtendSigned, ExtendUnsigned, TruncateInteger, SignedToFloat, UnsignedToFloat,
  FloatToSigned, FloatToUnsigned, ExtendFloat, TruncateFloat, IndexSigned,
  IndexUnsigned
};

enum class ScalarRelation { Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual };
enum class ScalarNaN { None, Ordered, Unordered, AlwaysFalse, AlwaysTrue };

struct ScalarComparison {
  ScalarRelation relation = ScalarRelation::Equal;
  ScalarNaN nan = ScalarNaN::None;
  bool unsignedInteger = false;
  bool relationResult = true;
};

struct ScalarOperation {
  mlir::Operation *operation;
  ScalarKind kind;
  ScalarComparison comparison;
  std::optional<ScalarCast> cast;
  mlir::Attribute constant;

  mlir::Type type() const { return operation->getResult(0).getType(); }
  mlir::Type inputType(unsigned index = 0) const {
    return operation->getOperand(index).getType();
  }
  static mlir::FailureOr<ScalarOperation> read(mlir::Operation *operation);
};

bool isStandardScalarOperation(mlir::Operation *operation);
unsigned scalarIntegerWidth(mlir::Type type);
llvm::StringRef scalarComparisonToken(ScalarRelation relation);
llvm::StringRef scalarComparisonMethod(ScalarRelation relation);

using ScalarRenderer = llvm::function_ref<mlir::FailureOr<std::string>(
    mlir::Operation *, llvm::ArrayRef<std::string>)>;
// Legality uses the actual expression renderer, without SSA lookup or output.
mlir::LogicalResult verifyScalarEmission(mlir::Operation *operation,
                                        ScalarRenderer renderer);

// Storage spelling is target-owned. The optional conversions express a
// floating storage carrier such as BANG C's uint16_t representation of bf16.
struct CScalarType {
  std::string name;
  std::string decode;
  std::string encode;
};
using CScalarTypes = llvm::function_ref<std::optional<CScalarType>(mlir::Type)>;
mlir::FailureOr<std::string> emitCScalar(const ScalarOperation &operation,
                                       llvm::ArrayRef<std::string> operands,
                                       CScalarTypes types);

} // namespace intent
#endif
