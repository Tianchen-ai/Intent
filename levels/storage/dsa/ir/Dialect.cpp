#include "compilerinfra/Assembly.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "Dialect.h.inc"
#include "Dialect.cpp.inc"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"
void intentdsl::dsa::SupplyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}
namespace CI_UNIT_NAMESPACE {
void dialect(mlir::DialectRegistry &r) {
  r.insert<intentdsl::dsa::SupplyDialect>();
}
} // namespace CI_UNIT_NAMESPACE
