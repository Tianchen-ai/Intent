compilerinfra_target(block levels/execution/block)
compilerinfra_target(kir levels/algorithm/kir)
compilerinfra_unit(SOURCES ir/Dialect.cpp ir/Implementations.cpp capabilities/Rows.cpp cost/Matrix.cpp legalize/Views.cpp emit/Kernel.cpp emit/Rows.cpp legalize/ArrayGeometry.cpp legalize/ArrayPanels.cpp legalize/ScaledPanels.cpp emit/Tiles.cpp
  LINK_LIBRARIES ${block} ${kir}
  TABLEGEN "ir/Tile.td Dialect.h.inc -gen-dialect-decls -dialect=ct_src"
           "ir/Tile.td Dialect.cpp.inc -gen-dialect-defs -dialect=ct_src"
           "ir/Tile.td Ops.h.inc -gen-op-decls" "ir/Tile.td Ops.cpp.inc -gen-op-defs")
