#ifndef INTENT_TRANSFORMS_COMPONENTS_H
#define INTENT_TRANSFORMS_COMPONENTS_H

#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/JSON.h"
#include <memory>

namespace intent {

/// Names are the normal MLIR pass arguments. The catalog describes scheduling
/// and controls; executable decisions remain in the operations rewritten by it.
struct TransformationComponent {
  llvm::StringRef owner;
  llvm::StringRef stage;
  bool required;
};

void registerTransformation(TransformationComponent component,
                            std::unique_ptr<mlir::Pass> pass);
void addTransformation(mlir::OpPassManager &manager,
                       TransformationComponent component,
                       std::unique_ptr<mlir::Pass> pass);
llvm::json::Array transformationComponents();
mlir::LogicalResult verifyTransformationControls(mlir::Operation *operation,
                                                CompileOptionsAttr options);

} // namespace intent
#endif
