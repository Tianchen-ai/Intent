#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace intent::cpu {
namespace {

LogicalResult verifyTransfer(Operation *operation, ShapedType source,
                             ShapedType destination) {
  if (source.getElementType() != destination.getElementType() ||
      failed(verifyCompatibleShape(source, destination)))
    return operation->emitOpError(
        "tensor value and memory must have matching elements and compatible shapes");
  return success();
}

} // namespace

LogicalResult ReadOp::verify() {
  return verifyTransfer(*this, cast<MemRefType>(getSource().getType()),
                        getResult().getType());
}

LogicalResult WriteOp::verify() {
  return verifyTransfer(*this, getSource().getType(),
                        cast<MemRefType>(getDestination().getType()));
}

} // namespace intent::cpu
