#pragma once
#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intentdsl::cpu {
struct FoldProducts {
  mlir::scf::ForOp loop;
  mlir::BlockArgument owner;
  int64_t depth, steps;
  bool prefix;
  llvm::SmallVector<mlir::Operation *> products;
  llvm::SmallVector<mlir::Operation *> boundOps;
  bool monotoneBound;
  bool nativeBf16;
  bool rectangular;
};

std::optional<FoldProducts> foldProducts(mlir::Operation &space,
                                         mlir::func::FuncOp fn,
                                         const CallFacts &facts);
bool fitsProductPanel(const FoldProducts &p, int64_t rows, int64_t keys,
                      int64_t extent, const RowMachine &m);
} // namespace intentdsl::cpu
