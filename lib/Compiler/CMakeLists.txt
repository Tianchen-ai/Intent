get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_library(MLIRIntentCompiler
  Compiler.cpp
  Backend.cpp
  GPUBackends.cpp
  CPUBackends.cpp
  DSABackends.cpp
  Passes.cpp
  Registration.cpp
  DEPENDS IntentCompilerPassIncGen
  LINK_LIBS PUBLIC
  ${dialect_libs}
  MLIRIntentKIRToGPU MLIRIntentKIRToCPU MLIRIntentKIRToDSA
  MLIRIntentDSATransforms MLIRIntentBangC
  MLIRIntentCPUAnalysis MLIRIntentCPUTransforms MLIRIntentMojoTransforms MLIRIntentMojoSerialization
  MLIRIntentGPUTransforms
  MLIRIntentTritonTransforms MLIRIntentTritonSerialization
  MLIRIntentCuTileTransforms MLIRIntentCuTileSerialization
  MLIRIntentTransforms MLIRFuncInlinerExtension MLIRPass MLIRTransforms MLIRIR MLIRSupport
)
if(TARGET MLIRIntentWeftTransforms)
  target_link_libraries(MLIRIntentCompiler PUBLIC MLIRIntentWeftTransforms MLIRIntentWeftSerialization)
  target_compile_definitions(MLIRIntentCompiler PUBLIC INTENT_HAS_WEFT_CANONICAL=1)
  target_compile_definitions(obj.MLIRIntentCompiler PRIVATE INTENT_HAS_WEFT_CANONICAL=1)
endif()
