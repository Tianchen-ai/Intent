add_mlir_dialect_library(MLIRIntentCuTileDialect
  CuTileDialect.cpp
  CuTileOps.cpp
  Configuration.cpp

  DEPENDS
  IntentCuTileIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRIntentBlockDialect
  MLIRIntentBlockAnalysis
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
  MLIRViewLikeInterface
  MLIRSupport
)
