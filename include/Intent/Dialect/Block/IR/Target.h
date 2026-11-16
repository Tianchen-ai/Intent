#ifndef INTENT_DIALECT_BLOCK_IR_TARGET_H
#define INTENT_DIALECT_BLOCK_IR_TARGET_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectInterface.h"
#include "llvm/Support/JSON.h"

namespace intent::block {
struct Resources {
  int64_t parallelUnits;
  int64_t valueBudgetWords;
  int64_t localStorageBudgetBytes;
  int64_t preferredVectorElements;
  int64_t doublePrecisionCost;
};

/// Target facts are typed attributes owned by the target dialect. These queries
/// expose only the budgets consumed by block transforms; zero means unknown.
class TargetInterface : public mlir::DialectInterface::Base<TargetInterface> {
public:
  explicit TargetInterface(mlir::Dialect *dialect) : Base(dialect) {}
  virtual bool supports(mlir::Attribute target) const = 0;
  virtual Resources resources(mlir::Attribute target) const = 0;
  virtual llvm::json::Object serialize(mlir::Attribute target) const = 0;
};

const TargetInterface *targetInterface(mlir::Attribute target);
Resources queryResources(mlir::Attribute target);
}
#endif
