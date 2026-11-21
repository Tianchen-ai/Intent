compilerinfra_target(cpu levels/execution/cpu)
compilerinfra_unit(SOURCES emit/Exit.cpp emit/Task.cpp ir/Host.cpp legalize/Panels.cpp LINK_LIBRARIES ${cpu}
 TABLEGEN "ir/Host.td Dialect.h.inc -gen-dialect-decls -dialect=weft_host"
 "ir/Host.td Dialect.cpp.inc -gen-dialect-defs -dialect=weft_host"
 "ir/Host.td Ops.h.inc -gen-op-decls" "ir/Host.td Ops.cpp.inc -gen-op-defs")
