#pragma once
#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace intentdsl::block {
// Prove that buckets own disjoint output intervals. The prefix must scan
// exactly the counts produced by the same key expression over the same row.
inline bool hasBucketPrefixes(mlir::scf::ForOp scatter, mlir::func::FuncOp fn,
                              const BoundedScatterLoopInfo &info) {
  using namespace mlir;
  auto cursor = info.cursorLoad;
  SmallVector<Value> protectedViews{cursor.getMemRef()};
  for (auto read : scatter.getBody()->getOps<memref::LoadOp>())
    protectedViews.push_back(read.getMemRef());
  auto previousLoop = [&](Operation *from) -> scf::ForOp {
    for (auto *op = from->getPrevNode(); op; op = op->getPrevNode()) {
      if (auto loop = dyn_cast<scf::ForOp>(op))
        return loop;
      if (isMemoryEffectFree(op) || isa<memref::DimOp>(op))
        continue;
      auto store = dyn_cast<memref::StoreOp>(op);
      if (!store || !llvm::all_of(protectedViews, [&](Value view) {
            return orderedViewsDisjoint(fn, nullptr, view, store.getMemRef());
          }))
        break;
    }
    return {};
  };
  auto prefix = previousLoop(scatter);
  ExclusiveScanLoopInfo scan;
  if (!prefix || !matchExclusiveScanLoop(prefix, fn, nullptr, &scan) ||
      scan.extent != info.bins ||
      !matchPattern(prefix.getInitArgs()[0], m_Zero()))
    return false;
  if (!llvm::any_of(scan.stores, [&](auto entry) {
        auto [store, inclusive] = entry;
        return !inclusive && store.getMemRef() == cursor.getMemRef() &&
               store.getIndices().drop_back() ==
                   cursor.getIndices().drop_back();
      }))
    return false;
  for (auto &entry : scan.stores)
    for (auto read : scatter.getBody()->getOps<memref::LoadOp>())
      if (read != cursor && !orderedViewsDisjoint(fn, nullptr, read.getMemRef(),
                                                  entry.first.getMemRef()))
        return false;
  protectedViews.push_back(scan.srcLoad.getMemRef());
  auto count = previousLoop(prefix);
  BoundedHistogramLoopInfo hist;
  if (!count || !matchBoundedHistogramLoop(count, fn, nullptr, &hist) ||
      hist.histStore.getMemRef() != scan.srcLoad.getMemRef() ||
      hist.histStore.getIndices().drop_back() !=
          scan.srcLoad.getIndices().drop_back() ||
      !matchPattern(hist.deltaVal, m_One()) ||
      orderedStaticLoopExtent(count, fn, nullptr) !=
          orderedStaticLoopExtent(scatter, fn, nullptr))
    return false;
  auto sameKey = [&](auto &self, Value a, Value b) -> bool {
    if (a == b ||
        (a == count.getInductionVar() && b == scatter.getInductionVar()))
      return true;
    auto *x = a.getDefiningOp(), *y = b.getDefiningOp();
    return x && y && x->getBlock() == count.getBody() &&
           y->getBlock() == scatter.getBody() && x->getName() == y->getName() &&
           x->getNumResults() == 1 && y->getNumResults() == 1 &&
           a.getType() == b.getType() &&
           x->getNumOperands() == y->getNumOperands() &&
           x->getAttrs() == y->getAttrs() &&
           x->getPropertiesAsAttribute() == y->getPropertiesAsAttribute() &&
           llvm::all_of(
               llvm::zip(x->getOperands(), y->getOperands()), [&](auto pair) {
                 return self(self, std::get<0>(pair), std::get<1>(pair));
               });
  };
  if (!sameKey(sameKey, hist.binIdx, info.binIdx))
    return false;
  auto clear = previousLoop(count);
  if (!clear || orderedStaticLoopExtent(clear, fn, nullptr) != info.bins ||
      !matchPattern(clear.getLowerBound(), m_Zero()) ||
      !matchPattern(clear.getStep(), m_One()))
    return false;
  auto stores = clear.getBody()->getOps<memref::StoreOp>();
  if (!llvm::hasSingleElement(stores))
    return false;
  auto zero = *stores.begin();
  return llvm::all_of(clear.getBody()->without_terminator(),
                      [&](Operation &op) {
                        return &op == zero.getOperation() ||
                               isMemoryEffectFree(&op);
                      }) &&
         zero.getMemRef() == hist.histStore.getMemRef() &&
         zero.getIndices().drop_back() ==
             hist.histStore.getIndices().drop_back() &&
         zero.getIndices().back() == clear.getInductionVar() &&
         matchPattern(zero.getValue(), m_Zero());
}
inline int64_t sortedBucketWidth(mlir::scf::ForOp loop, mlir::func::FuncOp fn,
                                 const BoundedScatterLoopInfo &scat,
                                 int64_t rankWidth, int64_t sortLimit) {
  auto extent = orderedStaticLoopExtent(loop, fn, nullptr);
  if (!extent || *extent <= rankWidth || *extent > INT32_MAX ||
      sortLimit <= rankWidth || scat.bins >= INT32_MAX / sortLimit)
    return 0;
  // The gather implementation must accept every payload, including tails.
  for (auto store : scat.scatterStores) {
    auto t = store.getValue().getType();
    if (!t.isInteger(32) && !t.isInteger(64) && !t.isF16() && !t.isBF16() &&
        !t.isF32())
      return 0;
  }
  return hasBucketPrefixes(loop, fn, scat)
             ? std::min<int64_t>(sortLimit, llvm::PowerOf2Ceil(*extent))
             : 0;
}
} // namespace intentdsl::block
