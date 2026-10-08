#ifndef INTENT_TARGET_WEFT_TRANSFORMS_PASSES_H
#define INTENT_TARGET_WEFT_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"

namespace intent::weft_provider {
#define GEN_PASS_DECL
#include "Intent/Target/Weft/Transforms/Passes.h.inc"

void registerWeftPasses();
void buildWeftPipeline(mlir::OpPassManager &manager);
cpu::ImplementationRegistry implementations();
mlir::LogicalResult legalizeProgram(mlir::ModuleOp program);
}
#endif
