add_mlir_library(MLIRIntentTritonTransforms
  Legalize.cpp
  Passes.cpp
  Access/AccessForms.cpp
  Access/GatherNormalization.cpp
  Access/GatherSelection.cpp
  Access/GatherStorage.cpp
  Access/Scatter.cpp
  Collective/Callbacks.cpp
  Collective/ScanTails.cpp
  Supply/Supply.cpp
  Value/Values.cpp
  Configuration/Configurations.cpp
  Configuration/ConfigurationFacts.cpp
  Mapping/ProgramGrid.cpp

  DEPENDS
  IntentTritonTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRIntentGPUAnalysis
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
  MLIRIntentTritonDialect
  MLIRIntentTritonAnalysis
  MLIRIntentTritonSerialization
)
