add_mlir_library(MLIRIntentConversionSupport
  ScalarLowering.cpp
  LogicalShape.cpp
  IndexedAccess.cpp
  LINK_LIBS PUBLIC MLIRIntentAnalysis MLIRIntentDialect MLIRArithDialect MLIRMathDialect MLIRDialectUtils MLIRIR)

add_subdirectory(KIRToGPU)
add_subdirectory(KIRToCPU)
add_subdirectory(KIRToDSA)
