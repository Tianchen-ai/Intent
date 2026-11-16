add_mlir_library(MLIRIntentKIRToBlock
  KIRToBlock.cpp
  KernelABI.cpp
  KernelConstruction.cpp
  Types.cpp
  RegionLowering.cpp
  Coordinates.cpp
  AccessCoordinates.cpp
  AccessRelations.cpp
  Memory.cpp
  Values.cpp
  Structured.cpp
  Quantization.cpp
  Control.cpp
  ConstructionSchema.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentAnalysis
  MLIRIntentConversionSupport
  MLIRIntentBlockDialect
  MLIRIntentBlockTransforms
  MLIRIntentDialect
  MLIRIntentTransforms
)
