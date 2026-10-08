add_mlir_library(MLIRIntentDSATransforms
  Passes.cpp
  MatrixSupply.cpp
  MatrixSupplyRelations.cpp
  MatrixPanels.cpp
  CollectiveGather.cpp
  IndexPrograms.cpp
  LocalSupplyRelations.cpp
  LocalValues.cpp
  LocalCopies.cpp
  LocalTransfers.cpp
  LocalMemoryValues.cpp
  Collective/Passes.cpp
  Collective/Realize.cpp
  Collective/Storage.cpp
  Collective/ScalarCombine.cpp
  Collective/NativeReduction.cpp

  DEPENDS
  IntentDSAPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIntentAnalysis
  MLIRIntentDSAAnalysis
  MLIRIntentDSADialect
  MLIRIntentDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
)
