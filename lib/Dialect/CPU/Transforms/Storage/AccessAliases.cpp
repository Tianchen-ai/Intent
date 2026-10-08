#include "AccessAliases.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace intent::cpu {

LogicalResult foldLoopAccessSubviews(scf::ForOp loop) {
  SmallVector<Operation *> accesses;
  loop.walk([&](Operation *operation) {
    if (isa<memref::LoadOp, memref::StoreOp>(operation) &&
        operation->getParentOfType<scf::ForOp>() == loop)
      accesses.push_back(operation);
  });
  llvm::SmallSetVector<Operation *, 32> affineIndices;
  IRRewriter rewriter(loop.getContext());
  for (Operation *access : accesses) {
    auto load = dyn_cast<memref::LoadOp>(access);
    auto store = dyn_cast<memref::StoreOp>(access);
    Value memory = load ? load.getMemref() : store.getMemref();
    auto originalView = memory.getDefiningOp<memref::SubViewOp>();
    if (!originalView || !loop->isAncestor(originalView)) continue;
    SmallVector<Value> indices(load ? load.getIndices() : store.getIndices());
    rewriter.setInsertionPoint(access);
    Operation *previous = access->getPrevNode();
    while (auto view = memory.getDefiningOp<memref::SubViewOp>()) {
      if (!loop->isAncestor(view)) break;
      SmallVector<Value> sourceIndices;
      affine::resolveIndicesIntoOpWithOffsetsAndStrides(
          rewriter, access->getLoc(), view.getMixedOffsets(), view.getMixedStrides(),
          view.getDroppedDims(), indices, sourceIndices);
      memory = view.getSource();
      indices = std::move(sourceIndices);
    }
    rewriter.modifyOpInPlace(access, [&] {
      if (load) {
        load.getMemrefMutable().assign(memory);
        load.getIndicesMutable().assign(indices);
      } else {
        store.getMemrefMutable().assign(memory);
        store.getIndicesMutable().assign(indices);
      }
    });
    // Legalize only index expressions inserted for these accesses. In
    // particular, do not rewrite invariant reshape or vector-access producers.
    for (Operation *created = previous ? previous->getNextNode() : &access->getBlock()->front();
         created != access; created = created->getNextNode())
      if (isa<affine::AffineApplyOp>(created)) affineIndices.insert(created);
    auto view = originalView;
    while (view && loop->isAncestor(view) && view->use_empty()) {
      auto source = view.getSource().getDefiningOp<memref::SubViewOp>();
      rewriter.eraseOp(view);
      view = source;
    }
  }
  if (affineIndices.empty()) return success();
  RewritePatternSet patterns(loop.getContext());
  populateAffineToStdConversionPatterns(patterns);
  GreedyRewriteConfig config;
  config.scope = &loop.getRegion();
  config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
  if (failed(applyOpPatternsGreedily(affineIndices.getArrayRef(), std::move(patterns), config)))
    return loop.emitError("loop access index composition did not converge");
  return success();
}

LogicalResult foldPrivateAccessAliases(func::FuncOp function) {
  llvm::SmallSetVector<Operation *, 32> accesses;
  {
    StorageAnalysis storage(function);
    function.walk([&](Operation *operation) {
      Value memory;
      if (auto load = dyn_cast<memref::LoadOp>(operation))
        memory = load.getMemref();
      else if (auto store = dyn_cast<memref::StoreOp>(operation))
        memory = store.getMemref();
      if (!memory) return;
      Value origin = storage.uniqueOrigin(memory);
      if (!origin || !origin.getDefiningOp<memref::AllocOp>() || memory == origin)
        return;

      // This is a private-intermediate optimization, not an ABI descriptor
      // normalization. Keep external views and vector access layout decisions
      // intact. Native patterns own offset/stride/rank-reduction composition.
      accesses.insert(operation);
      while (Operation *view = memory.getDefiningOp()) {
        if (!isa<memref::SubViewOp, memref::ExpandShapeOp,
                 memref::CollapseShapeOp, memref::CastOp>(view))
          break;
        accesses.insert(view);
        memory = view->getOperand(0);
      }
    });
  }
  if (accesses.empty()) return success();

  RewritePatternSet patterns(function.getContext());
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  // Native alias folding may create affine.apply. Close that implementation
  // detail here; CPU physical programs continue to contain ordinary arithmetic.
  populateAffineToStdConversionPatterns(patterns);
  GreedyRewriteConfig config;
  config.scope = &function.getBody();
  config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
  if (failed(applyOpPatternsGreedily(accesses.getArrayRef(),
                                    std::move(patterns), config)))
    return function.emitError("private access alias folding did not converge");
  return success();
}

} // namespace intent::cpu
