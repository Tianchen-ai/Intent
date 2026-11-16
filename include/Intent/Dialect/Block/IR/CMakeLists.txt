set(LLVM_TARGET_DEFINITIONS BlockDialect.td)
mlir_tablegen(BlockDialect.h.inc -gen-dialect-decls -dialect=intent_block)
mlir_tablegen(BlockDialect.cpp.inc -gen-dialect-defs -dialect=intent_block)

set(LLVM_TARGET_DEFINITIONS BlockAttrs.td)
mlir_tablegen(BlockAttrsEnums.h.inc -gen-enum-decls)
mlir_tablegen(BlockAttrsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(BlockAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(BlockAttrs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS BlockTypes.td)
mlir_tablegen(BlockTypes.h.inc -gen-typedef-decls)
mlir_tablegen(BlockTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS AccessOpInterface.td)
mlir_tablegen(AccessOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(AccessOpInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS FragmentOpInterface.td)
mlir_tablegen(FragmentOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(FragmentOpInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS BlockOps.td)
mlir_tablegen(BlockOps.h.inc -gen-op-decls)
mlir_tablegen(BlockOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(IntentBlockIncGen)
