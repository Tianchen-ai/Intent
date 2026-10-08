add_mlir_library(MLIRIntentStructuredOpInterface
  StructuredOpInterface.cpp
  DEPENDS IntentStructuredOpInterfaceIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSupport)
