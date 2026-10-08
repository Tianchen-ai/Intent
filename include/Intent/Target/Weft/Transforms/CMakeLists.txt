set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name WeftTransform)
add_public_tablegen_target(IntentWeftTransformsIncGen)
intent_pass_reference(weft)
