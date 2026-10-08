set(LLVM_TARGET_DEFINITIONS StructuredOpInterface.td)
mlir_tablegen(StructuredOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(StructuredOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(IntentStructuredOpInterfaceIncGen)
