set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name IntentCompiler)
add_public_tablegen_target(IntentCompilerPassIncGen)
intent_pass_reference(compiler)
