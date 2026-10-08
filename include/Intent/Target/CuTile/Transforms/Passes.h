#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_PASSES_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::cutile {

#define GEN_PASS_DECL
#include "Intent/Target/CuTile/Transforms/Passes.h.inc"

void registerCuTilePasses();
void buildCuTilePipeline(mlir::OpPassManager &manager);

} // namespace intent::cutile

#endif
