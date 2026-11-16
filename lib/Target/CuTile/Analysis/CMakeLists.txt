add_mlir_library(MLIRIntentCuTileAnalysis
  Tuning.cpp
  Program.cpp
  Configuration.cpp
  IndexBounds.cpp
  DEPENDS IntentNVIDIATargetIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRIntentCuTileDialect
  MLIRIntentBlockAnalysis
  MLIRIntentNVIDIATarget
)
