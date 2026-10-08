set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name IntentCuTileTransform)
add_public_tablegen_target(IntentCuTileTransformsIncGen)
intent_pass_reference(cutile)
