add_mlir_library(MLIRIntentBlockSerialization
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
  MLIRIntentBlockAnalysis
  MLIRIntentBlockDialect
)
