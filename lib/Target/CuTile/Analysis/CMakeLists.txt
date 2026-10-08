add_mlir_library(MLIRIntentCuTileAnalysis
  Tuning.cpp
  Program.cpp
  Configuration.cpp
  IndexBounds.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRIntentCuTileDialect
  MLIRIntentGPUAnalysis
)
