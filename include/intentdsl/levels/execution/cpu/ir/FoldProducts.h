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

// Seed each remaining row loop by operand position. Two carries can have
// distinct partial sums even when the source shares their zero initializer.
struct FoldRemainder {
  mlir::Value lower, upper;
  llvm::SmallVector<mlir::Value> initial;
};

std::optional<FoldProducts> foldProducts(mlir::Operation &space,
                                         mlir::func::FuncOp fn,
                                         const CallFacts &facts);
bool fitsProductPanel(const FoldProducts &p, int64_t rows, int64_t keys,
                      int64_t extent, const RowMachine &m);
} // namespace intentdsl::cpu
