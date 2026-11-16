set(LLVM_OPTIONAL_SOURCES Target.cpp)
add_mlir_library(MLIRIntentNVIDIATarget
  IR/Target.cpp
  DEPENDS IntentNVIDIATargetIncGen
  LINK_LIBS PUBLIC MLIRIntentBlockDialect MLIRIR MLIRSupport)
