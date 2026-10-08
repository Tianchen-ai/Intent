set(LLVM_TARGET_DEFINITIONS TritonDialect.td)
mlir_tablegen(TritonDialect.h.inc -gen-dialect-decls -dialect=intent_triton)
mlir_tablegen(TritonDialect.cpp.inc -gen-dialect-defs -dialect=intent_triton)

set(LLVM_TARGET_DEFINITIONS TritonOps.td)
mlir_tablegen(TritonOps.h.inc -gen-op-decls)
mlir_tablegen(TritonOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(IntentTritonIncGen)
