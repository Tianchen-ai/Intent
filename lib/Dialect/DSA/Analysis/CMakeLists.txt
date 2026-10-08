add_mlir_library(MLIRIntentDSAAnalysis
  PhysicalProgram.cpp
  Storage.cpp
  StorageQueries.cpp
  UniformValues.cpp

  LINK_LIBS PUBLIC
  MLIRIntentDSADialect
  MLIRIntentAnalysis
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
)
