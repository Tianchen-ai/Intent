#ifndef INTENT_DIALECT_DSA_IR_EXECUTIONRELATIONS_H
#define INTENT_DIALECT_DSA_IR_EXECUTIONRELATIONS_H

#include "Intent/Dialect/Intent/IR/Interface.h"
#include <memory>
#include <optional>

namespace mlir::scf {
class ForOp;
}

namespace intent::dsa {

enum class CoordinateDependency { Independent, Dependent, Unknown };

// One current-IR snapshot of scalar execution relations. Recreate after
// rewriting operands, regions, storage, or effects. This is also the query
// used by the program verifier; it does not own an execution plan.
class ExecutionRelations {
public:
  explicit ExecutionRelations(mlir::func::FuncOp function);
  // Query a proposed grouping of consecutive iterations. The only additional
  // uniform facts are quotients whose actual rewrite is returned below.
  ExecutionRelations(mlir::scf::ForOp taskLoop, int64_t groupWidth);
  ~ExecutionRelations();

  // False means unknown or varying, never a proof of variation.
  bool isUniform(mlir::Value value);
  bool hasUniformControl(mlir::Operation *operation);
  // value(group, lane) = value(group, 0) + lane * coefficient, in the
  // signed 64-bit address domain. Unknown relations return nullopt.
  std::optional<int64_t> participantCoefficient(mlir::Value value);
  // If value is a supported task quotient, return its divisor after grouping.
  // Consumers collect these facts before modifying the current program.
  std::optional<int64_t> groupedQuotientDivisor(mlir::Value value);
  intent::ViewType readonlyView(mlir::Value value);

  // Scalar SSA dependence on this exact coordinate, not group uniformity.
  // Effectful or unresolved forwarding values remain unknown.
  CoordinateDependency coordinateDependency(mlir::Value value,
                                           mlir::BlockArgument coordinate);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace intent::dsa

#endif
