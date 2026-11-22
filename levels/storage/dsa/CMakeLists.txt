compilerinfra_target(block levels/execution/block)
compilerinfra_unit(SOURCES cost/Supply.cpp ir/Dialect.cpp analysis/Verify.cpp analysis/Lifetime.cpp lowering/Stages.cpp decisions/Supply.cpp transforms/ReuseFragments.cpp
  LINK_LIBRARIES ${block}
  TABLEGEN "ir/Ops.td Dialect.h.inc -gen-dialect-decls -dialect=dsa"
           "ir/Ops.td Dialect.cpp.inc -gen-dialect-defs -dialect=dsa"
           "ir/Ops.td Ops.h.inc -gen-op-decls" "ir/Ops.td Ops.cpp.inc -gen-op-defs")
