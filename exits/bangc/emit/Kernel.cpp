#include "../legalize/Rows.h"
#include "intentdsl/levels/algorithm/kir/ir/Matrix.h"
#include "intentdsl/levels/storage/dsa/ir/Owners.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;

llvm::Error emitMatrix(func::FuncOp, llvm::raw_pwrite_stream &);
llvm::Error emitOrdered(func::FuncOp, llvm::raw_pwrite_stream &);

llvm::Error emit(ModuleOp m, llvm::raw_pwrite_stream &out) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  if (intentdsl::kir::matrixDomain(fn))
    return emitMatrix(fn, out);
  if (intentdsl::kir::orderedOwners(fn))
    return emitOrdered(fn, out);
  return rowPrinter().emit(m, out);
}
} // namespace CI_UNIT_NAMESPACE
