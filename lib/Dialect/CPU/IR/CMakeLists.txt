add_mlir_dialect_library(MLIRIntentCPUDialect
  CPUDialect.cpp
  CPUOps.cpp
  ValueOps.cpp
  CollectiveHelpers.cpp
  ShapeRelations.cpp
  CPUOpInterfaces.cpp
  DEPENDS IntentCPUIncGen
  LINK_LIBS PUBLIC
  MLIRIntentDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)
