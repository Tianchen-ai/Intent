set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BangC)
add_public_tablegen_target(IntentBangCPassesIncGen)
intent_pass_reference(bangc)
