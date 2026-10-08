#include "MemoryAccess.h"
#include "AccessAliases.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"

using namespace mlir;

namespace intent::cpu {

LogicalResult optimizeMemoryAccesses(func::FuncOp function) {
  if (function.isExternal()) return success();
  if (failed(foldPrivateAccessAliases(function))) return failure();
  bool changed;
  do {
    changed = reuseMemoryValues(function);
    changed |= detail::placeInvariantMemory(function);
    // Forwarding can expose invariant addresses. Placement in turn brings
    // observations into the same scope. Each step queries the resulting IR.
  } while (changed);
  eraseDeadPrivateBuffers(function);
  return success();
}

} // namespace intent::cpu
