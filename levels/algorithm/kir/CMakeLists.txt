compilerinfra_unit(SOURCES ir/Dialect.cpp ir/Matrix.cpp ir/Ordered.cpp lowering/Boundaries.cpp
  LINK_LIBRARIES MLIRFuncDialect MLIRArithDialect MLIRMemRefDialect MLIRSCFDialect MLIRMathDialect
  TABLEGEN
    "ir/Ops.td Dialect.h.inc -gen-dialect-decls -dialect=kir"
    "ir/Ops.td Dialect.cpp.inc -gen-dialect-defs -dialect=kir"
    "ir/Ops.td Ops.h.inc -gen-op-decls"
    "ir/Ops.td Ops.cpp.inc -gen-op-defs")
