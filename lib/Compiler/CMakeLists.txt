get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(backend_names GLOBAL PROPERTY INTENT_BACKEND_NAMES)
get_property(backend_sources GLOBAL PROPERTY INTENT_BACKEND_SOURCES)
get_property(backend_libraries GLOBAL PROPERTY INTENT_BACKEND_LIBRARIES)
set(backend_declarations "")
foreach(name IN LISTS backend_names)
  string(APPEND backend_declarations "INTENT_BACKEND(${name})\n")
endforeach()
file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/include/Intent/Compiler/Backends.inc"
  CONTENT "${backend_declarations}")
add_mlir_library(MLIRIntentCompiler
  Compiler.cpp
  Backend.cpp
  Models/Block.cpp
  Models/CPU.cpp
  Passes.cpp
  Registration.cpp
  ${backend_sources}
  ../Target/NVIDIA/Target.cpp
  DEPENDS IntentCompilerPassIncGen
  LINK_LIBS PUBLIC
  ${dialect_libs}
  MLIRIntentKIRToBlock MLIRIntentKIRToCPU MLIRIntentKIRToDSA
  MLIRIntentDSATransforms
  MLIRIntentCPUAnalysis MLIRIntentCPUTransforms
  MLIRIntentBlockTransforms
  ${backend_libraries}
  MLIRIntentTransforms MLIRFuncInlinerExtension MLIRPass MLIRTransforms MLIRIR MLIRSupport
)
if(TARGET MLIRIntentWeftTransforms)
  target_link_libraries(MLIRIntentCompiler PUBLIC MLIRIntentWeftTransforms MLIRIntentWeftSerialization)
  target_compile_definitions(MLIRIntentCompiler PUBLIC INTENT_HAS_WEFT_CANONICAL=1)
  target_compile_definitions(obj.MLIRIntentCompiler PRIVATE INTENT_HAS_WEFT_CANONICAL=1)
endif()
