add_mlir_dialect_library(MLIRIntentDSADialect
  DSADialect.cpp DSAOps.cpp Collectives.cpp Views.cpp MemoryEffects.cpp ExecutionRelations.cpp
  DEPENDS IntentDSAIncGen
  LINK_LIBS PUBLIC MLIRIntentDialect MLIRIntentAnalysis MLIRIR MLIRSupport MLIRSideEffectInterfaces
    MLIRBufferizationDialect MLIRBufferizationTransforms
    MLIRArithDialect MLIRFuncDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect)
