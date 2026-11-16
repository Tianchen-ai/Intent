#ifndef INTENT_TARGET_BANGC_CONSTRUCTION_PASSES_H
#define INTENT_TARGET_BANGC_CONSTRUCTION_PASSES_H
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
namespace intent::bangc {
#define GEN_PASS_DECL
#include "Intent/Target/DSA/BangC/Construction.h.inc"
void registerConstructionPasses();
}
#endif
