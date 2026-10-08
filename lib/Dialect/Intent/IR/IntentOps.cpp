#include "Intent/Dialect/Intent/IR/IntentOps.h"

using namespace mlir;

namespace intent {
LogicalResult ReturnOp::verify() { return success(); }
LogicalResult YieldOp::verify() { return success(); }
} // namespace intent

#define GET_OP_CLASSES
#include "Intent/Dialect/Intent/IR/IntentOps.cpp.inc"
