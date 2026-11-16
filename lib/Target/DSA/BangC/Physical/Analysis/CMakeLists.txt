add_mlir_library(MLIRIntentBangCPhysicalAnalysis
  PhysicalProgram.cpp
  Storage.cpp
  StorageQueries.cpp
  UniformValues.cpp

  LINK_LIBS PUBLIC
  MLIRIntentBangCPhysicalDialect
  MLIRIntentAnalysis
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
)
