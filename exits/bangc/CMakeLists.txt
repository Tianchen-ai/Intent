compilerinfra_target(kir levels/algorithm/kir)
compilerinfra_target(dsa levels/storage/dsa)
compilerinfra_unit(SOURCES
    ir/Dialect.cpp ir/Implementations.cpp transforms/NarrowAddress.cpp
    cost/Transfers.cpp cost/Matrix.cpp capabilities/Rows.cpp
    legalize/Bind.cpp legalize/Rows.cpp legalize/Matrix.cpp legalize/Ordered.cpp legalize/VerifyRows.cpp
    emit/Rows.cpp
  LINK_LIBRARIES ${dsa} ${kir}
  TABLEGEN
    "ir/Ops.td Dialect.h.inc -gen-dialect-decls -dialect=bangc_src"
    "ir/Ops.td Dialect.cpp.inc -gen-dialect-defs -dialect=bangc_src"
    "ir/Ops.td Ops.h.inc -gen-op-decls"
    "ir/Ops.td Ops.cpp.inc -gen-op-defs")
