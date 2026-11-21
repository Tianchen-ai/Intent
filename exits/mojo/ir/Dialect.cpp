#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "Dialect.h.inc"
#include "Dialect.cpp.inc"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"
void intentdsl::rowmojo::RowMojoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}
