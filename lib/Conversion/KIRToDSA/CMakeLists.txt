add_mlir_conversion_library(MLIRIntentKIRToDSA
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

  LINK_LIBS PUBLIC MLIRIntentAnalysis MLIRIntentConversionSupport MLIRIntentDialect MLIRIntentDSADialect
    MLIRIntentDSAAnalysis MLIRIntentDSATransforms MLIRIntentBangCResources
    MLIRArithDialect MLIRFuncDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect
    MLIRIR MLIRSupport MLIRTransformUtils)
