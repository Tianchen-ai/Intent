add_subdirectory(Physical)
add_subdirectory(Construction)
intent_backend(BangC SOURCE Backend.cpp SOURCES Compilation.cpp LIBRARIES MLIRIntentBangC MLIRIntentKIRToDSA)
add_mlir_library(MLIRIntentBangCResources
  NativeWorkspace.cpp
  LINK_LIBS PUBLIC MLIRIntentDialect MLIRIntentBangCPhysicalDialect MLIRIR MLIRSupport)

add_mlir_library(MLIRIntentBangC
  Transforms/Legalize.cpp
  Transforms/Matrix.cpp
  Transforms/MatrixInitialization.cpp
  Transforms/Numerics.cpp
  Transforms/BroadcastOperands.cpp
  Transforms/TransposeSupply.cpp
  Transforms/IndexLowering.cpp
  Transforms/Supply.cpp
  Transforms/CoarsenLocal.cpp
  Transforms/InvariantSupply.cpp
  Transforms/CompletedSupply.cpp
  Transforms/PipelineSupply.cpp
  Transforms/Storage.cpp
  Transforms/Verifier.cpp
  Serialization/Serializer.cpp
  Serialization/Scalar.cpp
  DEPENDS IntentBangCPassesIncGen
  LINK_LIBS PUBLIC MLIRIntentDialect MLIRIntentNativeABI MLIRIntentScalarSerialization MLIRIntentNativeSourceSerialization MLIRIntentBangCPhysicalDialect MLIRIntentBangCPhysicalAnalysis MLIRIntentBangCPhysicalTransforms MLIRArithDialect MLIRFuncDialect
    MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect MLIRTransforms MLIRPass MLIRIR MLIRSupport MLIRIntentBangCResources)
