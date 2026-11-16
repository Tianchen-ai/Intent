add_mlir_dialect_library(MLIRIntentBlockDialect
  Target.cpp
  BlockDialect.cpp
  ProgramInterface.cpp
  PhysicalExpressions.cpp
  IntegerRanges.cpp
  TypeVerification.cpp
  BlockOps.cpp
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
  IntentBlockIncGen
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
