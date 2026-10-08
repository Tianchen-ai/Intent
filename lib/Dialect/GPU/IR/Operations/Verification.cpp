#include "Verification.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
namespace intent::gpu::operation_detail {

Type elementType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  return type;
}

bool sameShape(Type lhs, Type rhs) {
  auto left = dyn_cast<FragmentType>(lhs);
  auto right = dyn_cast<FragmentType>(rhs);
  if (static_cast<bool>(left) != static_cast<bool>(right))
    return false;
  return !left || (left.getShape() == right.getShape() &&
                   left.getAxisMaps() == right.getAxisMaps() &&
                   left.getValidity() == right.getValidity() &&
                   left.getOwner() == right.getOwner());
}

LogicalResult verifyHelperRegion(Operation *owner, Region &region,
                                 TypeRange argumentTypes,
                                 TypeRange resultTypes) {
  if (!llvm::hasSingleElement(region))
    return owner->emitOpError("physical helper requires one block");
  Block &block = region.front();
  if (!llvm::equal(block.getArgumentTypes(), argumentTypes) || block.empty())
    return owner->emitOpError("physical helper arguments disagree with its schema");
  auto yield = dyn_cast<YieldOp>(block.back());
  if (!yield || !llvm::equal(yield.getOperandTypes(), resultTypes)) {
    InFlightDiagnostic diagnostic =
        owner->emitOpError("physical helper yield disagrees with its schema");
    diagnostic << "; expected=[";
    for (Type type : resultTypes)
      diagnostic << type << ", ";
    diagnostic << "]";
    if (yield) {
      diagnostic << "; actual=[";
      for (Type type : yield.getOperandTypes())
        diagnostic << type << ", ";
      diagnostic << "]";
    } else {
      diagnostic << "; actual=<no yield>";
    }
    return failure();
  }
  WalkResult effects = region.walk([&](Operation *nested) {
    if (isa<YieldOp>(nested))
      return WalkResult::advance();
    if (!isMemoryEffectFree(nested)) {
      nested->emitOpError("is effectful inside a physical pure helper");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return effects.wasInterrupted() ? failure() : success();
}

} // namespace intent::gpu::operation_detail
