add_mlir_dialect_library(MLIRIntentTritonDialect
  TritonDialect.cpp
  TritonOps.cpp
  Configuration.cpp
  Program.cpp

  DEPENDS
  IntentTritonIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentBlockDialect
  MLIRIntentBlockAnalysis
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
  MLIRViewLikeInterface
  MLIRSCFDialect
  MLIRSupport
)
