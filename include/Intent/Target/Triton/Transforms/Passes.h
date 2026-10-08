#ifndef INTENT_TARGET_TRITON_TRANSFORMS_PASSES_H
#define INTENT_TARGET_TRITON_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
namespace intent::triton {

#define GEN_PASS_DECL
#include "Intent/Target/Triton/Transforms/Passes.h.inc"

void registerTritonPasses();
void buildTritonPipeline(mlir::OpPassManager &manager);

} // namespace intent::triton

#endif
