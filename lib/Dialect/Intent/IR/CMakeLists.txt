add_mlir_dialect_library(MLIRIntentDialect
  IntentDialect.cpp
  Interface.cpp
  CompileOptions.cpp
  IntentOps.cpp
  TypeSchema.cpp
  Access.cpp
  ValueOps.cpp
  ShapeOps.cpp
  StructuredOps.cpp
  ControlFlow.cpp

  DEPENDS
  IntentIncGen
  IntentStructuredOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRIntentStructuredOpInterface
  MLIRPresburger
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
  MLIRSupport
)
