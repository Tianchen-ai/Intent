#ifndef INTENT_DSA_TRANSFORMS_PASS_SUPPORT_H
#define INTENT_DSA_TRANSFORMS_PASS_SUPPORT_H

#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace intent::dsa::detail {

inline mlir::LogicalResult finishTransform(mlir::ModuleOp module,
                                           llvm::StringRef name,
                                           mlir::LogicalResult result) {
  if (mlir::failed(result))
    return module.emitError() << "DSA transformation failed: " << name;
  if (mlir::failed(verifyRealizedProgram(module)))
    return module.emitError() << "DSA postcondition failed: " << name;
  return mlir::success();
}

} // namespace intent::dsa::detail

#endif
