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
  Option<int64_t> vectorBits{*this, "vector-bits", llvm::cl::init(0)};
  Option<int64_t> workers{*this, "workers", llvm::cl::init(0)};
  Option<int64_t> privateBytes{*this, "private-bytes", llvm::cl::init(262144)};
  Option<bool> matrixI8I32{*this, "matrix-i8-i32", llvm::cl::init(false)};
};

void buildCPUPipeline(mlir::OpPassManager &manager, const CPUCompilationOptions &options);

void registerCPUPasses();

}

#endif
