set(LLVM_TARGET_DEFINITIONS CuTileDialect.td)
mlir_tablegen(CuTileDialect.h.inc -gen-dialect-decls -dialect=intent_cutile)
mlir_tablegen(CuTileDialect.cpp.inc -gen-dialect-defs -dialect=intent_cutile)

set(LLVM_TARGET_DEFINITIONS CuTileOps.td)
mlir_tablegen(CuTileOps.h.inc -gen-op-decls)
mlir_tablegen(CuTileOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(IntentCuTileIncGen)
