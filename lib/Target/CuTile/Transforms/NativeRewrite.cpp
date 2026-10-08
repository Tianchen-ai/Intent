#include "NativeRewrite.h"
#include <cassert>

using namespace mlir;

namespace intent::cutile {

Type withElementType(Type type, Type elementType) {
  auto fragment = dyn_cast<gpu::FragmentType>(type);
  if (!fragment)
    return elementType;
  return gpu::FragmentType::get(
      fragment.getContext(), elementType, fragment.getShape(),
      fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
}

void NativeFormRewriter::replace(Operation *operation, ValueRange values) {
  assert(operation->getNumResults() == values.size());
  replacements.push_back({operation, llvm::to_vector(values)});
}
void NativeFormRewriter::erase(Operation *operation) {
  assert(operation->use_empty());
  replacements.push_back({operation, {}});
}
void NativeFormRewriter::commit() {
  for (Replacement &replacement : replacements)
    if (!replacement.values.empty())
      replacement.operation->getResults().replaceAllUsesWith(replacement.values);
  // Destroy inner operations before their owners even if a future native form
  // consumes nested input operations in a different construction order.
  auto depth = [](Operation *operation) {
    unsigned result = 0;
    while ((operation = operation->getParentOp())) ++result;
    return result;
  };
  llvm::stable_sort(replacements, [&](const Replacement &a, const Replacement &b) {
    return depth(a.operation) > depth(b.operation);
  });
  for (Replacement &replacement : replacements)
    replacement.operation->erase();
  replacements.clear();
}

} // namespace intent::cutile
