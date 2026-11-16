set(LLVM_TARGET_DEFINITIONS ExecutionInterfaces.td)
mlir_tablegen(ExecutionInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ExecutionInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(IntentDSAExecutionInterfacesIncGen)
