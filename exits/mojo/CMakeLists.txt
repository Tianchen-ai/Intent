compilerinfra_target(cpu levels/execution/cpu)
compilerinfra_unit(SOURCES capabilities/OneDNN.cpp emit/Exit.cpp emit/Source.cpp emit/Host.cpp emit/Matrix.cpp emit/Ordered.cpp cost/Matrix.cpp legalize/Loops.cpp legalize/ABI.cpp ir/Dialect.cpp LINK_LIBRARIES ${cpu}
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
