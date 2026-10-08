add_mlir_library(MLIRIntentGPUSerialization
  Interface.cpp
  Python.cpp
  PythonEmitter.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentSourceSerialization
  MLIRIntentGPUAnalysis
  MLIRIntentGPUDialect
)
