set(LLVM_OPTIONAL_SOURCES Backend.cpp)
add_mlir_library(MLIRIntentAscendTarget
  IR/Target.cpp
  DEPENDS IntentAscendTargetIncGen
  LINK_LIBS PUBLIC MLIRIntentBlockDialect MLIRIR MLIRSupport)
intent_backend(TritonAscend SOURCE Backend.cpp
  PROFILE Configuration/TuningProfiles.json PROFILE_NAME triton_ascend.json
  LIBRARIES MLIRIntentAscendTarget MLIRIntentTritonTransforms MLIRIntentTritonSerialization)
