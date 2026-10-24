add_mlir_library(MLIRIntentMojoTransforms
  Legalize.cpp
  ComputeScopes.cpp
  Passes.cpp
  Implementations.cpp
  MaterializeRegisterContractions.cpp
  Matrix.cpp
  Quantization.cpp
  DEPENDS IntentMojoTransformsIncGen
  LINK_LIBS PUBLIC
  MLIRIntentCPUTransforms
  MLIRIntentMojoSerialization
  MLIRPass
  MLIRArithDialect
  MLIRAMXDialect
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
