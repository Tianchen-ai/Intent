#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "Intent/Target/Triton/IR/TritonOps.h"

using namespace mlir;

#define GET_DIALECT_DEF
#include "Intent/Target/Triton/IR/TritonDialect.cpp.inc"

namespace intent::triton {

void IntentTritonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Intent/Target/Triton/IR/TritonOps.cpp.inc"
      >();
}

} // namespace intent::triton
