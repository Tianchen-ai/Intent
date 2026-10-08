add_mlir_library(MLIRIntentTransforms
  VerifyKernelIR.cpp
  NormalizeKernelIR.cpp

  LINK_LIBS PUBLIC
  MLIRIntentAnalysis
  MLIRIntentDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
