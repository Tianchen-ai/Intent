set(LLVM_TARGET_DEFINITIONS GPUDialect.td)
mlir_tablegen(GPUDialect.h.inc -gen-dialect-decls -dialect=intent_gpu)
mlir_tablegen(GPUDialect.cpp.inc -gen-dialect-defs -dialect=intent_gpu)

set(LLVM_TARGET_DEFINITIONS GPUAttrs.td)
mlir_tablegen(GPUAttrsEnums.h.inc -gen-enum-decls)
mlir_tablegen(GPUAttrsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(GPUAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(GPUAttrs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS GPUTypes.td)
mlir_tablegen(GPUTypes.h.inc -gen-typedef-decls)
mlir_tablegen(GPUTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS AccessOpInterface.td)
mlir_tablegen(AccessOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(AccessOpInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS FragmentOpInterface.td)
mlir_tablegen(FragmentOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(FragmentOpInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS GPUOps.td)
mlir_tablegen(GPUOps.h.inc -gen-op-decls)
mlir_tablegen(GPUOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(IntentGPUIncGen)
