add_mlir_library(MLIRIntentTritonSerialization
  Serializer.cpp
  Numerical.cpp
  DEPENDS IntentNVIDIATargetIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentBlockDialect
  MLIRIntentBlockSerialization
  MLIRIntentTritonDialect
  MLIRIntentTritonAnalysis
)
