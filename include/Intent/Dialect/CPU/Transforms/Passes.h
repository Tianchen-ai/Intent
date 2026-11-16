#ifndef INTENT_DIALECT_CPU_TRANSFORMS_PASSES_H
#define INTENT_DIALECT_CPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace intent::cpu {

#define GEN_PASS_DECL
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

struct CPUCompilationOptions : mlir::PassPipelineOptions<CPUCompilationOptions> {
  Option<std::string> provider{*this, "provider", llvm::cl::desc("Compiled CPU source provider")};
  Option<std::string> defaults{*this, "defaults", llvm::cl::desc("Provider default tuning profile file")};
  Option<std::string> overrides{*this, "overrides", llvm::cl::desc("Tuning profile overrides file")};
  Option<std::string> target{*this, "target", llvm::cl::desc("Typed CPU capabilities and execution configuration")};
};

void buildCPUValuePipeline(mlir::OpPassManager &manager, const CPUCompilationOptions &options);
/// Requires the CPU Buffers stage. A component chooses when to materialize
/// storage; this reusable sequence does not silently convert tensor SSA.
void buildCPUBufferPipeline(mlir::OpPassManager &manager, const CPUCompilationOptions &options);

void registerCPUPasses();

}

#endif
