set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name IntentGPUTransform)
add_public_tablegen_target(IntentGPUTransformsIncGen)
intent_pass_reference(gpu)
