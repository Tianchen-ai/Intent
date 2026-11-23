#pragma once
#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
namespace intentdsl::block {
struct TileDot {
  mlir::Value rows, tokens;
};
struct TileAccumulate {
  mlir::Value values, weights;
};
std::optional<TileDot> tileDot(mlir::Operation *, mlir::scf::ForOp);
std::optional<TileAccumulate> tileAccumulate(mlir::Value, mlir::scf::ForOp);
std::optional<unsigned> issueAxis(mlir::Operation *,
                                  const compilerinfra::TargetProfile &, int64_t,
                                  int64_t);
bool fullFoldPanels(mlir::scf::ForOp,const CallFacts &,int64_t span);
bool sharedBounds(mlir::scf::ForOp, unsigned, int64_t);
int64_t alignedPageExtent(mlir::Operation &, mlir::scf::ForOp,
                          const CallFacts &, int64_t);
mlir::Value causalPrefixOwner(mlir::scf::ForOp);
void partitionOwnerFolds(mlir::Operation *, unsigned, int64_t rows, int64_t span);
} // namespace intentdsl::block
