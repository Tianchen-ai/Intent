find_path(WEFT_KERNEL_INCLUDE_DIR Weft/Dialect/Kernel/IR/KernelDialect.h
  PATHS "${INTENT_WEFT_SOURCE_DIR}/include" NO_DEFAULT_PATH REQUIRED)
find_path(WEFT_KERNEL_GENERATED_INCLUDE_DIR Weft/Dialect/Kernel/IR/KernelOps.h.inc
  PATHS "${INTENT_WEFT_BINARY_DIR}/include" NO_DEFAULT_PATH REQUIRED)
find_library(WEFT_KERNEL_LIBRARY WeftKernelDialect
  PATHS "${INTENT_WEFT_BINARY_DIR}/lib/Dialect/Kernel/IR" NO_DEFAULT_PATH REQUIRED)
add_library(IntentWeftCanonical STATIC IMPORTED)
set_target_properties(IntentWeftCanonical PROPERTIES
  IMPORTED_LOCATION "${WEFT_KERNEL_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${WEFT_KERNEL_INCLUDE_DIR};${WEFT_KERNEL_GENERATED_INCLUDE_DIR}")
add_subdirectory(IR)
add_subdirectory(Transforms)
add_subdirectory(Serialization)
