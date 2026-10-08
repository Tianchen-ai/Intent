add_mlir_dialect_library(MLIRIntentGPUDialect
  GPUDialect.cpp
  ProgramInterface.cpp
  PhysicalExpressions.cpp
  IntegerRanges.cpp
  TypeVerification.cpp
  GPUOps.cpp
  Operations/Program.cpp
  Operations/Values.cpp
  Operations/Shape.cpp
  Operations/Access.cpp
  Operations/Contraction.cpp
  Operations/Collective.cpp
  Operations/Verification.cpp
  Canonicalization.cpp
  AccessOpInterface.cpp
  FragmentOpInterface.cpp

  DEPENDS
  IntentGPUIncGen
  IntentStructuredOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIntentDialect
  MLIRIntentAnalysis
  MLIRInferIntRangeInterface
  MLIRIntentStructuredOpInterface
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)
