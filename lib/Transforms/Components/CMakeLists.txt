add_mlir_library(MLIRIntentTransformComponents
  Components.cpp
  LINK_LIBS PUBLIC MLIRIntentDialect MLIRIR MLIRPass MLIRSupport
)
