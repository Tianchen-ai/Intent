#ifndef INTENT_COMPILER_PASSES_H
#define INTENT_COMPILER_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace intent::compiler {
#define GEN_PASS_DECL
#include "Intent/Compiler/Passes.h.inc"
void registerCompilationPasses();
} // namespace intent::compiler
#endif
