add_mlir_library(MLIRIntentMojoTransforms
  Legalize.cpp
  Passes.cpp
  Implementations.cpp
  MaterializeRegisterContractions.cpp
  Quantization.cpp
  DEPENDS IntentMojoTransformsIncGen
  LINK_LIBS PUBLIC
  MLIRIntentCPUTransforms
  MLIRIntentMojoSerialization
  MLIRPass
  MLIRArithDialect
  MLIRArithTransforms
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRLinalgDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRIR
)
