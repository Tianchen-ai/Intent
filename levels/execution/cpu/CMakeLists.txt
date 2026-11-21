compilerinfra_target(kir levels/algorithm/kir)
compilerinfra_unit(SOURCES decisions/RowTasks.cpp ir/Dialect.cpp lowering/Boundaries.cpp
  LINK_LIBRARIES ${kir}
  TABLEGEN
    "ir/Ops.td Dialect.h.inc -gen-dialect-decls -dialect=cpu"
    "ir/Ops.td Dialect.cpp.inc -gen-dialect-defs -dialect=cpu"
    "ir/Ops.td Ops.h.inc -gen-op-decls"
    "ir/Ops.td Ops.cpp.inc -gen-op-defs")
