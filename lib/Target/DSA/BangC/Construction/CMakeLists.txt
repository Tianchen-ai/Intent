add_mlir_conversion_library(MLIRIntentKIRToDSA
  Passes.cpp
  KIRToDSA.cpp
  Construction.cpp
  Values.cpp
  Tensors.cpp
  Access.cpp
  AccessCoordinates.cpp
  IndexedUpdates.cpp
  Quantization.cpp
  Contractions.cpp
  ScaledContractions.cpp
  Collectives.cpp
  CollectiveConstruction.cpp
  Regions.cpp
  Worksets.cpp
  WorksetCohorts.cpp
  DEPENDS IntentBangCConstructionIncGen
  LINK_LIBS PUBLIC MLIRIntentAnalysis MLIRIntentConversionSupport MLIRIntentDialect MLIRIntentBangCPhysicalDialect
    MLIRIntentBangCPhysicalAnalysis MLIRIntentBangCPhysicalTransforms MLIRIntentBangCResources
    MLIRArithDialect MLIRFuncDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect
    MLIRIR MLIRSupport MLIRTransformUtils)
