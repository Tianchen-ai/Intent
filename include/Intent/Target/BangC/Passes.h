#ifndef INTENT_TARGET_BANGC_PASSES_H
#define INTENT_TARGET_BANGC_PASSES_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <string>
namespace mlir { class OpPassManager; }
namespace intent::bangc {
#define GEN_PASS_DECL
#include "Intent/Target/BangC/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "Intent/Target/BangC/Passes.h.inc"

void buildBangCPipeline(mlir::OpPassManager &manager, llvm::StringRef architecture);
void registerBangCPipelines();
mlir::LogicalResult verifyProgram(mlir::ModuleOp module);
mlir::LogicalResult serializeProgram(mlir::ModuleOp module, std::string &source, std::string &metadata);
}
#endif
