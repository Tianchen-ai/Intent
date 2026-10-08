set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MojoTransform)
add_public_tablegen_target(IntentMojoTransformsIncGen)
intent_pass_reference(mojo)
