add_mlir_library(MLIRIntentMojoScalarSerialization
  Scalar.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRIntentScalarSerialization MLIRIntentSourceSerialization
)

add_mlir_library(MLIRIntentMojoSerialization
  Serializer.cpp
  Matrix.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIntentCPUAnalysis
  MLIRIntentCPUDialect
  MLIRIntentNativeABI
  MLIRIntentNativeSourceSerialization
  MLIRIntentMojoScalarSerialization
  MLIRArithDialect
  MLIRAMXDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRTransformUtils
  MLIRIR
)
