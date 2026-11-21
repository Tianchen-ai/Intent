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
inline std::string tileLoop(int width) {
  std::string step = std::to_string(width);
  return "int64_t nt=(cols-1)/" + step + "+1, mt=(rows-1)/" + step +
         "+1;\n#pragma omp parallel for schedule(static) num_threads(workers) "
         "if(workers>1)\nfor(int64_t g=0;g<mt*nt;++g){int64_t r=(g/nt)*" +
         step + ", c=(g%nt)*" + step + "; int64_t nr=rows-r<" + step +
         "?rows-r:" + step + ", nc=cols-c<" + step + "?cols-c:" + step + ";";
}
} // namespace intentdsl::cpu
