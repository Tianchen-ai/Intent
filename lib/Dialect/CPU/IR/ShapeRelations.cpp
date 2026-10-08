#include "Intent/Dialect/CPU/IR/ShapeRelations.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace intent::cpu {

std::optional<PublicDimension> queryPublicDimension(BlockArgument argument,
                                                   int64_t axis) {
  if (!argument) return std::nullopt;
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.front() ||
      !function->hasAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr))
    return std::nullopt;
  auto interface = intent::getPublicInterface(function);
  if (!interface || interface.getArguments().size() != function.getNumArguments())
    return std::nullopt;
  auto view = intent::getPublicView(interface, argument.getArgNumber());
  if (!view || axis < 0 || axis >= intent::publicViewTensor(view).getRank())
    return std::nullopt;
  auto tensor = intent::publicViewTensor(view);
  PublicDimension result{argument, axis, std::nullopt};
  if (!tensor.isDynamicDim(axis)) result.constant = tensor.getDimSize(axis);
  int64_t identity = intent::publicViewDimensions(view)[axis];
  if (!identity) return result;
  bool found = false;
  for (auto [ordinal, unused] : llvm::enumerate(interface.getArguments())) {
    auto candidate = intent::getPublicView(interface, ordinal);
    if (!candidate) continue;
    auto shape = intent::publicViewTensor(candidate);
    for (auto [dimension, id] :
         llvm::enumerate(intent::publicViewDimensions(candidate).asArrayRef())) {
      if (id != identity) continue;
      if (!found) {
        result.argument = function.getArgument(ordinal);
        result.axis = dimension;
        found = true;
      }
      if (!shape.isDynamicDim(dimension))
        result.constant = shape.getDimSize(dimension);
    }
  }
  return result;
}

} // namespace intent::cpu
