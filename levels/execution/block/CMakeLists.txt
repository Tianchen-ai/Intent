compilerinfra_target(parent levels/algorithm/kir)
file(GLOB strategies CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "decisions/strategies/*.cpp")
set(declarations "")
set(calls "")
foreach(strategy IN LISTS strategies)
  get_filename_component(pattern "${strategy}" NAME_WE)
  string(APPEND declarations "void register${pattern}(compilerinfra::Unit &);\n")
  string(APPEND calls "  register${pattern}(unit);\n")
endforeach()
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/Strategies.inc"
     CONTENT "@declarations@\nstatic void registerStrategies(compilerinfra::Unit &unit) {\n@calls@}\n" @ONLY)
compilerinfra_unit(SOURCES cost/Tiles.cpp ir/Dialect.cpp analysis/Verify.cpp lowering/Boundaries.cpp ${strategies} transforms/Realize.cpp transforms/Normalize.cpp
  LINK_LIBRARIES ${parent}
  TABLEGEN
    "ir/Ops.td Dialect.h.inc -gen-dialect-decls -dialect=block"
    "ir/Ops.td Dialect.cpp.inc -gen-dialect-defs -dialect=block"
    "ir/Ops.td Ops.h.inc -gen-op-decls"
    "ir/Ops.td Ops.cpp.inc -gen-op-defs")
