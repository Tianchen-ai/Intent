add_mlir_dialect_library(MLIRIntentTritonDialect
  TritonDialect.cpp
  TritonOps.cpp
  Configuration.cpp
  Program.cpp

  DEPENDS
  IntentTritonIncGen
  IntentNVIDIATargetIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentBlockDialect
  MLIRIntentNVIDIATarget
  MLIRIntentBlockAnalysis
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
  MLIRViewLikeInterface
  MLIRSCFDialect
  MLIRSupport
)
