#include "Reductions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::weft_provider {

FailureOr<NativeReduction> queryNativeReduction(
    Operation *owner, Block &body, Value accumulator,
    cpu::ReductionOrderAttr order) {
  auto yielded = body.getTerminator()->getOperands();
  Operation *combine = yielded.size() == 1 ? yielded.front().getDefiningOp() : nullptr;
  if (!combine || combine->getBlock() != &body || !accumulator.hasOneUse() ||
      combine->getNumOperands() != 2 ||
      !llvm::is_contained(combine->getOperands(), accumulator))
    return owner->emitError("Weft reduction requires a closed accumulator combine"),
           failure();
  for (Operation &operation : body.without_terminator())
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation) ||
        !llvm::all_of(operation.getOperandTypes(), [](Type type) {
          return isa<FloatType, IntegerType, IndexType>(type);
        }))
      return owner->emitError("Weft native reduction requires a pure scalar contribution"),
             failure();
  StringRef kind;
  if (isa<arith::AddFOp, arith::AddIOp>(combine)) kind = "add";
  else if (isa<arith::MaximumFOp>(combine)) kind = "maximum";
  else if (isa<arith::MinimumFOp>(combine)) kind = "minimum";
  else if (isa<arith::MaxNumFOp>(combine)) kind = "max";
  else if (isa<arith::MinNumFOp>(combine)) kind = "min";
  else if (isa<arith::OrIOp>(combine) && accumulator.getType().isInteger(1)) kind = "or";
  else if (isa<arith::AndIOp>(combine) && accumulator.getType().isInteger(1)) kind = "and";
  else
    return owner->emitError("Weft reduction combine is not implemented"), failure();
  if (order && !order.getAdjacentReassociation())
    return owner->emitError("Weft native reduction requires reassociation permission"),
           failure();
  if (kind == "add" && isa<FloatType>(accumulator.getType()) &&
      (!order || !order.getElementPermutation()))
    return owner->emitError("Weft native floating-add reduction requires element-permutation permission"),
           failure();
  Value contribution = combine->getOperand(
      combine->getOperand(0) == accumulator ? 1 : 0);
  return NativeReduction{combine, contribution, kind};
}

} // namespace intent::weft_provider
