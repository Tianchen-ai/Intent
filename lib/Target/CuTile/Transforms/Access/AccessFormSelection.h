#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_ACCESSFORMSELECTION_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_ACCESSFORMSELECTION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cutile {

// Lazy bindings are shared by all access forms in a single native phase.
class AccessFormSelection {
public:
  explicit AccessFormSelection(mlir::func::FuncOp kernel) : kernel(kernel) {}
  mlir::FailureOr<mlir::Value> loadFormCondition();
  mlir::FailureOr<mlir::Value> tmaCondition();
  mlir::FailureOr<mlir::Value> loadPolicyValue();

private:
  mlir::FailureOr<mlir::Value> accessFormValue();
  mlir::func::FuncOp kernel;
  mlir::Value accessForm;
  mlir::Value loadPolicy;
  mlir::Value preferTileLoads;
  mlir::Value allowNativeTMA;
};

} // namespace intent::cutile

#endif
