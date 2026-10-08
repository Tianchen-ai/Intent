add_mlir_library(MLIRIntentNativeABI
  NativeABI.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRIntentDialect MLIRFuncDialect MLIRIR MLIRSupport
)

add_mlir_library(MLIRIntentScalarSerialization
  Scalar.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRArithDialect MLIRMathDialect MLIRIR MLIRSupport
)

add_mlir_library(MLIRIntentSourceSerialization
  Source.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRIR MLIRSupport
)

add_mlir_library(MLIRIntentNativeSourceSerialization
  NativeSource.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRIntentSourceSerialization MLIRMemRefDialect MLIRSCFDialect MLIRIR MLIRSupport
)
