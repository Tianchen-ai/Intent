compilerinfra_target(cpu levels/execution/cpu)
compilerinfra_target(algorithm levels/algorithm/kir)
compilerinfra_unit(SOURCES
  capabilities/OneDNN.cpp
  cost/Matrix.cpp
  emit/Exit.cpp
  emit/Host.cpp
  emit/MatrixPrinter.cpp
  emit/Ordered.cpp
  emit/Source.cpp
  ir/Dialect.cpp
  ir/MatrixContracts.cpp
  legalize/ABI.cpp
  legalize/Loops.cpp
  legalize/Matrix.cpp
  transforms/NarrowAddress.cpp
  LINK_LIBRARIES ${cpu} ${algorithm}
  TABLEGEN "ir/Loop.td Dialect.h.inc -gen-dialect-decls -dialect=mojo_row"
  "ir/Loop.td Dialect.cpp.inc -gen-dialect-defs -dialect=mojo_row"
  "ir/Loop.td Ops.h.inc -gen-op-decls" "ir/Loop.td Ops.cpp.inc -gen-op-defs")
if(CI_OPTIONAL_onednn)
  get_target_property(ci_onednn_inc ${CI_OPTIONAL_onednn} INTERFACE_INCLUDE_DIRECTORIES)
  if(ci_onednn_inc)
    target_include_directories(${ci_target} PRIVATE ${ci_onednn_inc})
  endif()
  target_compile_definitions(${ci_target} PRIVATE
    CI_ONEDNN_HEADER_AVAILABLE=1
    CI_ONEDNN_LIBRARY_PATH="$<TARGET_FILE:${CI_OPTIONAL_onednn}>")
endif()
