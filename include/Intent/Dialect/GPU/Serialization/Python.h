#ifndef INTENT_DIALECT_GPU_SERIALIZATION_PYTHON_H
#define INTENT_DIALECT_GPU_SERIALIZATION_PYTHON_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace intent::gpu {

// A terminal source projection, not another interface schema. Parameter
// identity and legality are owned by ProgramInterface; this only groups the
// current arguments and assigns names in the generated Python scope.
struct PythonArgument {
  mlir::BlockArgument value;
  std::string name;

  ViewType viewType() const { return mlir::cast<ViewType>(value.getType()); }
};

struct PythonSignature {
  llvm::SmallVector<PythonArgument> views;
  llvm::SmallVector<PythonArgument> scalars;
  llvm::SmallVector<PythonArgument> metadata;

  static mlir::FailureOr<PythonSignature> read(mlir::func::FuncOp kernel);
};

struct PythonScalarSyntax {
  llvm::StringRef prefix;
  llvm::StringRef boolean;
  // An empty spelling preserves a provider's unsupported scalar type.
  llvm::StringRef float64;
  llvm::StringRef float8E4M3FN;
  llvm::StringRef float8E5M2;
};

std::string pythonScalarType(mlir::Type type, const PythonScalarSyntax &syntax);

struct PythonExpressionSyntax {
  // Runtime division follows the provider's integer semantics. Constant
  // expressions use Python's mathematical floor/ceil without fixed-width sums.
  std::string (*integerDivision)(PhysicalExprKind, llvm::StringRef,
                                 llvm::StringRef);
  llvm::StringRef minimum;
  llvm::StringRef maximum;
  // An empty select name requests an ordinary Python conditional expression.
  llvm::StringRef select;
  llvm::StringRef nextPowerOfTwo;
  bool clampNextPowerOfTwo;
};

bool isConstexprPhysicalExpression(PhysicalExprAttr expression);

std::string pythonExpression(
    PhysicalExprAttr expression, const PythonExpressionSyntax &syntax,
    llvm::function_ref<std::string(PhysicalExprAttr)> symbol);

// The callback only spells a positive infinity; this function preserves its
// sign and owns bool/integer, finite floating-point and NaN spelling.
std::string pythonLiteral(
    mlir::Attribute value,
    llvm::function_ref<std::string(mlir::Type)> infinity = nullptr);

} // namespace intent::gpu
#endif
