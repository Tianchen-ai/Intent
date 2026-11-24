compilerinfra_target(cpu levels/execution/cpu)
compilerinfra_target(algorithm levels/algorithm/kir)
compilerinfra_unit(SOURCES
  cost/Matrix.cpp
  emit/Exit.cpp
  emit/MatrixPrinter.cpp
  ir/Host.cpp
  ir/MatrixContracts.cpp
  legalize/Matrix.cpp
  legalize/Ordered.cpp
  legalize/Rows.cpp
  transforms/NarrowAddress.cpp
  LINK_LIBRARIES ${cpu} ${algorithm}
  TABLEGEN "ir/Host.td Dialect.h.inc -gen-dialect-decls -dialect=weft_host"
  "ir/Host.td Dialect.cpp.inc -gen-dialect-defs -dialect=weft_host"
  "ir/Host.td Ops.h.inc -gen-op-decls" "ir/Host.td Ops.cpp.inc -gen-op-defs")
