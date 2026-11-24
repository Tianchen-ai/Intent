compilerinfra_target(kir levels/algorithm/kir)
compilerinfra_unit(SOURCES decisions/MatrixBuffers.cpp decisions/Matrix.cpp cost/Ordered.cpp decisions/Ordered.cpp cost/Contraction.cpp cost/Vector.cpp cost/Traversal.cpp decisions/Pointwise.cpp decisions/RowTasks.cpp decisions/RowReduction.cpp decisions/TupleReduction.cpp decisions/PanelReduction.cpp decisions/Scan.cpp decisions/Indirect.cpp decisions/Segment.cpp decisions/KeyReduction.cpp decisions/PrefixScatter.cpp decisions/Atomics.cpp ir/Dialect.cpp lowering/Boundaries.cpp
  decisions/OwnerTraversal.cpp
  LINK_LIBRARIES ${kir}
  TABLEGEN
    "ir/Ops.td Dialect.h.inc -gen-dialect-decls -dialect=cpu"
    "ir/Ops.td Dialect.cpp.inc -gen-dialect-defs -dialect=cpu"
    "ir/Ops.td Ops.h.inc -gen-op-decls"
    "ir/Ops.td Ops.cpp.inc -gen-op-defs")
