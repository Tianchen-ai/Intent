add_mlir_library(MLIRIntentWeftHostSource
  HostScalar.cpp
  HostSource.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRIntentCPUDialect MLIRIntentScalarSerialization
  MLIRIntentNativeSourceSerialization MLIRFuncDialect MLIRMemRefDialect
  MLIRSCFDialect MLIRIR
)

add_mlir_library(MLIRIntentWeftSerialization
  Serializer.cpp
  HostSerializer.cpp
  TaskABI.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC IntentWeftCanonical MLIRIntentWeftDialect MLIRIntentCPUDialect MLIRIntentNativeABI MLIRArithDialect
  MLIRFuncDialect MLIRMemRefDialect MLIRSCFDialect MLIRIR
)
