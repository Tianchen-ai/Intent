add_mlir_library(MLIRIntentDSAExecution
  IR/ExecutionInterfaces.cpp
  IR/Resources.cpp
  DEPENDS IntentDSAExecutionInterfacesIncGen
  LINK_LIBS PUBLIC MLIRSCFDialect MLIRSideEffectInterfaces MLIRIR MLIRSupport)
