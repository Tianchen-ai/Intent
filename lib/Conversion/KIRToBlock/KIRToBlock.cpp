#include "Construction.h"
#include "Intent/Conversion/KIRToBlock/KIRToBlock.h"
#include "Intent/Dialect/Block/Transforms/Passes.h"
#include "Intent/Transforms/Passes.h"

using namespace mlir;

namespace intent {

LogicalResult lowerCanonicalKIRToBlock(
    ModuleOp module,
    const BlockCapabilities &capabilities) {
  if (failed(verifyKernelModule(module)))
    return failure();
  if (!block::targetInterface(capabilities.target))
    return module.emitError("selected block target has no resource interface");
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1)
    return module.emitError("Block construction requires exactly one kernel entry");
  if (failed(kir_to_block::constructBlockProgram(module, capabilities, functions.front())))
    return failure();
  return block::completeBlockProgramConstruction(module);
}

} // namespace intent
