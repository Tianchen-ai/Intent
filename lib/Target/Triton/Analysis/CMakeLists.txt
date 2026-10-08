add_mlir_library(MLIRIntentTritonAnalysis
  Program.cpp
  Configuration.cpp
  Contractions.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUAnalysis
  MLIRIntentGPUDialect
  MLIRIntentTritonDialect
)
