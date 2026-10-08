#include "Accesses.h"
#include "AccessFormSelection.h"

using namespace mlir;

namespace intent::cutile {

LogicalResult formNativeAccesses(func::FuncOp kernel,
    const NativeAccessInputs &inputs,
    bool matrixCompute, NativeFormRewriter &rewriter) {
  AccessFormSelection forms(kernel);
  if (failed(formNativeLoads(kernel, inputs.loads, matrixCompute, forms, rewriter)) ||
      failed(formFragmentExtractions(kernel, inputs.gathers, rewriter)) ||
      failed(formNativeAtomics(kernel, inputs.atomics, rewriter)) ||
      failed(formNativeStores(kernel, inputs.stores, forms, rewriter)))
    return failure();
  return success();
}

} // namespace intent::cutile
