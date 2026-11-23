#pragma once
#include "intentdsl/levels/execution/block/ir/ScorePanel.h"
namespace intentdsl::dsa {
struct ScoreSupply {
  int64_t m, n, d, v;
  llvm::SmallVector<int64_t> sizes() const {
    // Packed operands, transposed V, fp32 logits and probabilities,
    // narrowed probabilities, weighted output, publication, row scratch
    // and two temporary panels for native arithmetic range checks.
    return {2 * m * d,
            2 * n * d,
            2 * n * v,
            2 * n * v,
            4 * m * n,
            4 * m * n,
            2 * m * n,
            4 * m * v,
            2 * m * v,
            4 * n,
            std::max(4 * m * std::min<int64_t>(64, std::max(n, v)),
                     2 * std::max({m * d, n * d, n * v, m * n})),
            2 * std::max({m * d, n * d, n * v, m * n})};
  }
  int64_t localBytes() const {
    int64_t total = 0;
    for (auto bytes : sizes())
      total += llvm::alignTo(bytes, 128);
    return total;
  }
  int64_t filterBytes() const { return 2 * n * std::max(d, v); }
};
inline ScoreSupply scoreSupply(mlir::Operation &space) {
  auto &score = space.getRegion(0).front().front();
  auto fn = space.getParentOfType<mlir::func::FuncOp>();
  auto shape = [&](unsigned i, int a) {
    return fn.getArgAttrOfType<mlir::DenseI64ArrayAttr>(i, "block.shape")[a];
  };
  return {
      mlir::cast<mlir::ShapedType>(score.getResult(0).getType()).getDimSize(0),
      shape(1, 2), shape(0, 3), shape(2, 3)};
}
} // namespace intentdsl::dsa
