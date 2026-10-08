add_mlir_library(MLIRIntentCuTileSerialization
  Serializer.cpp
  Numerical.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentCuTileDialect
  MLIRIntentCuTileAnalysis
  MLIRIntentGPUDialect
  MLIRIntentGPUSerialization
)
