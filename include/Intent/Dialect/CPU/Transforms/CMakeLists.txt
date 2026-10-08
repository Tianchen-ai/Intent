set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CPUTransform)
add_public_tablegen_target(IntentCPUTransformsIncGen)
intent_pass_reference(cpu)
