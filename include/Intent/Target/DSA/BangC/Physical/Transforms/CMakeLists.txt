set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name DSA)
add_public_tablegen_target(IntentBangCPhysicalPassesIncGen)
intent_pass_reference(dsa)
