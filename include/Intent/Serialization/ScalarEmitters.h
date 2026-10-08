#ifndef INTENT_SERIALIZATION_SCALAR_EMITTERS_H
#define INTENT_SERIALIZATION_SCALAR_EMITTERS_H

#include "Intent/Serialization/Scalar.h"
#include "Intent/Serialization/Source.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace intent {

// Context-specific emission uses the same typed surface as semantic decoding.
// Renderer must be an owning callable (normally a function pointer), as the
// operation table outlives this registration call.
template <typename Context, typename Renderer, typename Emit>
void addScalarEmitters(OperationEmitters<Context> &table, Renderer renderer,
                       Emit emit) {
  auto check = [renderer](mlir::Operation *operation) {
    return verifyScalarEmission(operation, renderer);
  };
#define INTENT_SCALAR_OP(Op, Kind, Cast) table.template add<Op>(check, emit);
#include "Intent/Serialization/ScalarOps.def"
#undef INTENT_SCALAR_OP
}

struct ScalarEmission {
  llvm::ArrayRef<std::string> operands;
  std::string expression;
};

template <typename Renderer>
OperationEmitters<ScalarEmission> scalarExpressionEmitters(Renderer renderer) {
  OperationEmitters<ScalarEmission> result;
  addScalarEmitters(result, renderer,
      [renderer](mlir::Operation *operation, ScalarEmission &context) {
        auto expression = renderer(operation, context.operands);
        if (mlir::failed(expression)) return mlir::failure();
        context.expression = std::move(*expression);
        return mlir::success();
      });
  return result;
}

inline mlir::FailureOr<std::string> emitScalarExpression(
    const OperationEmitters<ScalarEmission> &table, mlir::Operation *operation,
    llvm::ArrayRef<std::string> operands) {
  ScalarEmission context{operands, {}};
  if (mlir::failed(table.emit(operation, context))) return mlir::failure();
  return std::move(context.expression);
}

} // namespace intent
#endif
