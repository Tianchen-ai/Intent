#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl {
// Rectangular value computations with no coordinate-dependent effects may
// be traversed in either order. Per-buffer offsets are obtained from SSA.
inline bool rectangularCopy(mlir::Operation *op) {
  auto fn = op->getParentOfType<mlir::func::FuncOp>();
  if (!fn || fn->hasAttr("kir.iteration") || fn.getNumArguments() != 2)
    return false;
  for (auto arg : fn.getArguments()) {
    auto t = mlir::dyn_cast<mlir::MemRefType>(arg.getType());
    if (!t || t.getRank() != 2)
      return false;
  }
  auto &body = op->getRegion(0).front();
  if (body.getOperations().size() != 3)
    return false;
  auto &r = body.front();
  auto &w = *std::next(body.begin());
  return r.getName().stripDialect() == "read" &&
         w.getName().stripDialect() == "write" && w.getNumOperands() == 3 &&
         w.getOperand(2) == r.getResult(0) &&
         r.getOperand(0) == fn.getArgument(0) &&
         w.getOperand(0) == fn.getArgument(1) &&
         w.getAttrOfType<mlir::DenseI64ArrayAttr>("axes") &&
         w.getAttrOfType<mlir::DenseI64ArrayAttr>("axes").asArrayRef() ==
             llvm::ArrayRef<int64_t>({1, 0});
}
} // namespace intentdsl
