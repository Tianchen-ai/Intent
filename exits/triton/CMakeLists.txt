compilerinfra_target(parent levels/execution/block)
compilerinfra_target(algorithm levels/algorithm/kir)
compilerinfra_unit(SOURCES
    ir/Dialect.cpp ir/Implementations.cpp
    capabilities/Rows.cpp cost/Matrix.cpp transforms/NarrowAddress.cpp
    legalize/Legalize.cpp
    legalize/Rows.cpp legalize/VerifyTiles.cpp
    emit/Exit.cpp emit/Rows.cpp emit/Columns.cpp emit/Tiles.cpp
  LINK_LIBRARIES ${parent} ${algorithm}
  TABLEGEN
    "ir/Ops.td Dialect.h.inc -gen-dialect-decls -dialect=triton_src"
    "ir/Ops.td Dialect.cpp.inc -gen-dialect-defs -dialect=triton_src"
    "ir/Ops.td Ops.h.inc -gen-op-decls"
    "ir/Ops.td Ops.cpp.inc -gen-op-defs")
