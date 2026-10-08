set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name IntentTritonTransform)
add_public_tablegen_target(IntentTritonTransformsIncGen)
intent_pass_reference(triton)
