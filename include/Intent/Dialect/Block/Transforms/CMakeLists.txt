set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name IntentBlockTransform)
add_public_tablegen_target(IntentBlockTransformsIncGen)
intent_pass_reference(block)
