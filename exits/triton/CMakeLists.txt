compilerinfra_target(parent levels/execution/block)
compilerinfra_unit(SOURCES ir/Dialect.cpp emit/Exit.cpp legalize/Legalize.cpp capabilities/Rows.cpp
  LINK_LIBRARIES ${parent}
  TABLEGEN
    "ir/Ops.td Dialect.h.inc -gen-dialect-decls -dialect=triton_src"
    "ir/Ops.td Dialect.cpp.inc -gen-dialect-defs -dialect=triton_src"
    "ir/Ops.td Ops.h.inc -gen-op-decls"
    "ir/Ops.td Ops.cpp.inc -gen-op-defs")
