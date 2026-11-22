compilerinfra_target(cpu levels/execution/cpu)
compilerinfra_unit(SOURCES emit/Exit.cpp emit/Source.cpp emit/Host.cpp emit/Matrix.cpp cost/Matrix.cpp legalize/Loops.cpp legalize/ABI.cpp ir/Dialect.cpp LINK_LIBRARIES ${cpu}
 TABLEGEN "ir/Loop.td Dialect.h.inc -gen-dialect-decls -dialect=mojo_row"
 "ir/Loop.td Dialect.cpp.inc -gen-dialect-defs -dialect=mojo_row"
 "ir/Loop.td Ops.h.inc -gen-op-decls" "ir/Loop.td Ops.cpp.inc -gen-op-defs")
