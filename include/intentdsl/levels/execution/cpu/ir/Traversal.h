#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Permutation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::cpu {
using intentdsl::rectangularCopy;
inline mlir::DictionaryAttr partition(mlir::Builder &b, llvm::StringRef walk,
                                      mlir::ArrayAttr rows) {
  return b.getDictionaryAttr(
      {b.getNamedAttr("walk", b.getStringAttr(walk)),
       b.getNamedAttr("rows", rows),
       b.getNamedAttr("tile", b.getI64IntegerAttr(walk == "tile" ? 32 : 1))});
}
} // namespace intentdsl::cpu
