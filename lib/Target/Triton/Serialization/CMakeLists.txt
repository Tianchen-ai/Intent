add_mlir_library(MLIRIntentTritonSerialization
  Serializer.cpp
  Numerical.cpp

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
