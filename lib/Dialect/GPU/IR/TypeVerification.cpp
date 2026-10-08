#include "Intent/Dialect/GPU/IR/TypeVerification.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace intent::gpu {

LogicalResult verifyGPUTypeInvariants(Operation *root) {
  // AttrTypeWalker visits subelements before their owner and caches visited
  // attribute/type identities. A malformed child is diagnosed before a parent
  // verifier follows its typed operands; shared types are checked once.
  AttrTypeWalker walker;
  Operation *owner = root;
  auto diagnostic = [&] {
    return owner->emitError("GPU type/attribute invariant failed: ");
  };
  auto result = [](LogicalResult valid) {
    return failed(valid) ? WalkResult::interrupt() : WalkResult::advance();
  };
  walker.addWalk([&](ParameterRefAttr value) {
    return result(ParameterRefAttr::verify(diagnostic, value.getName()));
  });
  walker.addWalk([&](ArgumentRefAttr value) {
    return result(ArgumentRefAttr::verify(diagnostic, value.getId()));
  });
  walker.addWalk([&](PhysicalExprAttr value) {
    return result(PhysicalExprAttr::verify(diagnostic, value.getKind(),
        value.getValue(), value.getSymbol(), value.getOperands()));
  });
  walker.addWalk([&](AxisMapAttr value) {
    return result(AxisMapAttr::verify(diagnostic, value.getSourceId(),
        value.getSourceAxis(), value.getDimensionId(), value.getFragmentAxis(),
        value.getDerived()));
  });
  walker.addWalk([&](ViewLayoutAttr value) {
    return result(ViewLayoutAttr::verify(diagnostic, value.getExtents(),
        value.getDimensionIds(), value.getStrides()));
  });
  walker.addWalk([&](FragmentType value) {
    return result(FragmentType::verify(diagnostic, value.getElementType(),
        value.getShape(), value.getAxisMaps(), value.getValidity(), value.getOwner()));
  });
  walker.addWalk([&](ViewType value) {
    return result(ViewType::verify(diagnostic, value.getElementType(),
        value.getAccess(), value.getSourceId(), value.getLayout()));
  });
  walker.addWalk([&](BufferType value) {
    return result(BufferType::verify(diagnostic, value.getElementType(),
        value.getShape(), value.getScope(), value.getInstance(), value.getOwner(),
        value.getInitialization(), value.getVisibility()));
  });
  walker.addWalk([&](RecordType value) {
    return result(RecordType::verify(diagnostic, value.getFieldNames(),
        value.getFieldTypes(), value.getOwner()));
  });
  walker.addWalk([&](RangeType value) {
    return result(RangeType::verify(diagnostic, value.getSourceId(),
        value.getSourceAxis(), value.getDimensionId(), value.getDerived()));
  });
  WalkResult checked = root->walk<WalkOrder::PreOrder>([&](Operation *operation) {
    owner = operation;
    if (walker.walk(operation->getAttrDictionary()).wasInterrupted())
      return WalkResult::interrupt();
    for (Type type : operation->getOperandTypes())
      if (walker.walk(type).wasInterrupted()) return WalkResult::interrupt();
    for (Type type : operation->getResultTypes())
      if (walker.walk(type).wasInterrupted()) return WalkResult::interrupt();
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (walker.walk(argument.getType()).wasInterrupted())
            return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(checked.wasInterrupted());
}

} // namespace intent::gpu
