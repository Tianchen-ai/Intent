compilerinfra_target(block levels/execution/block)
compilerinfra_unit(SOURCES ir/Dialect.cpp capabilities/Rows.cpp cost/Matrix.cpp legalize/Views.cpp legalize/Owners.cpp emit/Kernel.cpp emit/Matrix.cpp
  LINK_LIBRARIES ${block}
  TABLEGEN "ir/Tile.td Dialect.h.inc -gen-dialect-decls -dialect=ct_src"
           "ir/Tile.td Dialect.cpp.inc -gen-dialect-defs -dialect=ct_src"
           "ir/Tile.td Ops.h.inc -gen-op-decls" "ir/Tile.td Ops.cpp.inc -gen-op-defs")
