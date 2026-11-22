#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "compilerinfra/Assembly.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "Dialect.h.inc"
#include "Dialect.cpp.inc"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"
void intentdsl::cutile::TileSource::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}
namespace CI_UNIT_NAMESPACE {
void registerIR(mlir::DialectRegistry &r) {
  r.insert<intentdsl::cutile::TileSource>();
}
} // namespace CI_UNIT_NAMESPACE
