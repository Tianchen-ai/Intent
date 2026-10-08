#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"

using namespace mlir;

#define GET_DIALECT_DEF
#include "Intent/Target/CuTile/IR/CuTileDialect.cpp.inc"

namespace intent::cutile {

void IntentCuTileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Intent/Target/CuTile/IR/CuTileOps.cpp.inc"
      >();
}

} // namespace intent::cutile
