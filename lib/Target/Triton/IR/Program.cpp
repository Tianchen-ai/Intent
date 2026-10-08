#include "Intent/Target/Triton/IR/Program.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::triton {

LogicalResult verifyProgramAttributes(Operation *operation) {
  if (Attribute attribute = operation->getAttr(contractFormAttr)) {
    auto form = dyn_cast<StringAttr>(attribute);
    if (!isa<gpu::ContractOp>(operation) || !form ||
        (form.getValue() != "multiply_sum" && form.getValue() != "fma"))
      return operation->emitOpError(
          "Triton contraction form requires a contract operation and a "
          "multiply_sum or fma string");
  }
  if (Attribute attribute = operation->getAttr(loopUnrollFactorAttr)) {
    auto factor = dyn_cast<IntegerAttr>(attribute);
    if (!isa<scf::ForOp>(operation) || !factor ||
        !factor.getType().isSignlessInteger(64) || factor.getInt() <= 0)
      return operation->emitOpError(
          "Triton loop unroll factor requires an scf.for and a positive i64");
  }
  return success();
}

} // namespace intent::triton
