#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "AccessAliases.h"
#include "../Vector/ProducerVectorization.h"
#include "../Structure/ProducerReuse.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerReplay.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

using namespace mlir;

namespace intent::cpu {
namespace {

bool sameStaticLayout(MemRefType source, MemRefType destination) {
  if (source.getShape() != destination.getShape() ||
      source.getElementType() != destination.getElementType() ||
      source.getMemorySpace() != destination.getMemorySpace()) return false;
  SmallVector<int64_t> sourceStrides, destinationStrides;
  int64_t sourceOffset, destinationOffset;
  if (failed(source.getStridesAndOffset(sourceStrides, sourceOffset)) ||
      failed(destination.getStridesAndOffset(destinationStrides, destinationOffset)) ||
      ShapedType::isDynamic(sourceOffset) || ShapedType::isDynamic(destinationOffset) ||
      sourceOffset != destinationOffset || sourceStrides != destinationStrides ||
      llvm::any_of(sourceStrides, ShapedType::isDynamic)) return false;
  return memref::CastOp::areCastCompatible(TypeRange{source}, TypeRange{destination});
}

void forwardDestinations(func::FuncOp function) {
  SmallVector<memref::CopyOp> copies;
  function.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    Value source = copy.getSource();
    Operation *reshape = source.getDefiningOp();
    if (unitReshapeAxes(reshape)) {
      if (!source.hasOneUse() || reshape->getBlock() != copy->getBlock()) continue;
      source = reshape->getOperand(0);
    } else reshape = nullptr;
    auto allocation = source.getDefiningOp<memref::AllocOp>();
    if (!allocation || allocation->getBlock() != copy->getBlock()) continue;
    Value target = copy.getTarget();
    MemRefType replacementType = cast<MemRefType>(copy.getTarget().getType());
    SmallVector<ReassociationIndices> reassociation;
    if (auto collapse = dyn_cast_or_null<memref::CollapseShapeOp>(reshape)) {
      reassociation = collapse.getReassociationIndices();
      auto expanded = memref::ExpandShapeOp::computeExpandedType(
          replacementType, allocation.getType().getShape(), reassociation);
      if (failed(expanded)) continue;
      replacementType = *expanded;
    } else if (auto expand = dyn_cast_or_null<memref::ExpandShapeOp>(reshape)) {
      reassociation = expand.getReassociationIndices();
      if (!memref::CollapseShapeOp::isGuaranteedCollapsible(replacementType, reassociation)) continue;
      replacementType = memref::CollapseShapeOp::computeCollapsedType(replacementType, reassociation);
    }
    if (allocation.getType().getRank() != replacementType.getRank()) continue;
    if (reshape && (replacementType.getShape() != allocation.getType().getShape() ||
                    replacementType.getMemorySpace() != allocation.getType().getMemorySpace())) continue;
    bool castReplacement = replacementType != allocation.getType() &&
        sameStaticLayout(replacementType, allocation.getType());
    StorageAnalysis storage(function);
    Value targetRoot = storage.uniqueOrigin(target);
    if (!targetRoot) continue;
    auto external = storage.externalView(target);
    if (!targetRoot.getDefiningOp<memref::AllocOp>() &&
        (!external || external.getAccess() != 1)) continue;
    if (targetRoot == allocation.getResult()) continue;
    auto targetOp = target.getDefiningOp();
    DominanceInfo dominance(function);
    if (targetOp && !dominance.dominates(targetOp, allocation)) {
      if (targetOp->getBlock() != allocation->getBlock() ||
          (!isMemoryEffectFree(targetOp) && !isa<memref::AllocOp>(targetOp)) ||
          llvm::any_of(targetOp->getOperands(), [&](Value value) {
            return !dominance.dominates(value, allocation);
          })) continue;
    }
    auto lifetime = storage.lifetime(allocation);
    if (!lifetime || !lifetime->aliases.complete) continue;
    bool legal = true;
    Operation *lastUse = copy;
    for (Operation *user : lifetime->aliases.users) {
      if (user == reshape || user == lifetime->end) continue;
      bool view = isStorageAliasOperation(user);
      // Retained views encode the private allocation's representation, including
      // reassociation and extracted metadata. Preserve their input type with a
      // proven static-layout cast; unknown strides cannot establish equivalence.
      // The copy-source reshape is removed separately.
      if (view && replacementType != allocation.getType() && !castReplacement) {
        legal = false;
        break;
      }
      if (!view && !isa<memref::CopyOp, memref::DimOp, memref::LoadOp, memref::StoreOp,
                       linalg::LinalgOp, ReduceOp, ScanOp, HistogramOp, QuantizedDotOp>(user)) {
        legal = false;
        break;
      }
      Operation *ancestor = copy->getBlock()->findAncestorOpInBlock(*user);
      if (!ancestor) { legal = false; break; }
      if (ancestor != copy && copy->isBeforeInBlock(ancestor)) {
        if (!view && !isa<memref::LoadOp, memref::DimOp>(user)) { legal = false; break; }
        if (lastUse->isBeforeInBlock(ancestor)) lastUse = ancestor;
      }
    }
    // Forwarding moves the destination writes earlier. Its previous contents
    // must remain unobserved before the copy and stable through all later reads.
    for (Operation *between = allocation->getNextNode(); legal && between != lastUse->getNextNode();
         between = between->getNextNode()) {
      if (between == copy) continue;
      auto effects = storage.effects(between);
      if (!effects.complete || effects.ordered) {
        legal = false;
        break;
      }
      for (const StorageEffect &entry : effects.entries) {
        const auto &effect = entry.effect;
        if (isa<MemoryEffects::Allocate>(effect.getEffect())) continue;
        if (!effect.getValue() || !storage.disjoint(effect.getValue(), target))
          legal = false;
      }
    }
    if (!legal) continue;
    if (targetOp && !dominance.dominates(targetOp, allocation)) targetOp->moveBefore(allocation);
    if (reshape) {
      OpBuilder builder(allocation);
      if (isa<memref::CollapseShapeOp>(reshape)) {
        SmallVector<OpFoldResult> shape;
        unsigned dynamicAxis = 0;
        for (int64_t size : allocation.getType().getShape())
          shape.push_back(ShapedType::isDynamic(size)
              ? OpFoldResult(allocation.getDynamicSizes()[dynamicAxis++])
              : OpFoldResult(builder.getIndexAttr(size)));
        target = builder.create<memref::ExpandShapeOp>(allocation.getLoc(), replacementType,
            target, reassociation, shape);
      } else {
        target = builder.create<memref::CollapseShapeOp>(allocation.getLoc(), replacementType,
            target, reassociation);
      }
    }
    if (castReplacement) {
      OpBuilder builder(allocation);
      target = builder.create<memref::CastOp>(allocation.getLoc(), allocation.getType(), target);
    }
    lifetime->end.erase();
    copy.erase();
    if (reshape) reshape->erase();
    allocation.getResult().replaceAllUsesWith(target);
    allocation.erase();
  }
}

bool fullVectorRead(vector::LoadOp load, memref::AllocOp allocation, Value &vectorLimit) {
  auto type = allocation.getType();
  auto vector = load.getVectorType();
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!type.getRank() || !type.getLayout().isIdentity() || vector.getRank() != 1 || vector.isScalable() ||
      failed(type.getStridesAndOffset(strides, offset)) || strides.back() != 1) return false;
  unsigned dynamicAxis = 0;
  for (int64_t axis = 0; axis + 1 < type.getRank(); ++axis) {
    int64_t size = type.getDimSize(axis);
    Value extent = type.isDynamicDim(axis) ? allocation.getDynamicSizes()[dynamicAxis++] : Value{};
    Value position = load.getIndices()[axis];
    if (auto constant = getConstantIntValue(position)) {
      if (extent || *constant < 0 || *constant >= size) return false;
      continue;
    }
    auto coordinate = dyn_cast<BlockArgument>(position);
    auto traversal = coordinate ? dyn_cast<scf::ForOp>(coordinate.getOwner()->getParentOp()) : scf::ForOp{};
    auto step = traversal ? getConstantIntValue(traversal.getStep()) : std::nullopt;
    auto begin = traversal ? getConstantIntValue(traversal.getLowerBound()) : std::nullopt;
    if (!traversal || position != traversal.getInductionVar() || !step || *step <= 0 ||
        !begin || *begin < 0) return false;
    auto end = getConstantIntValue(traversal.getUpperBound());
    if (extent ? traversal.getUpperBound() != extent : !end || *end > size) return false;
  }
  int64_t width = vector.getDimSize(0);
  int64_t staticExtent = type.getShape().back();
  Value extent;
  if (ShapedType::isDynamic(staticExtent)) extent = allocation.getDynamicSizes().back();
  auto isExtent = [&](Value value) {
    auto constant = getConstantIntValue(value);
    return extent ? value == extent : constant && *constant == staticExtent;
  };
  Value begin = load.getIndices().back();
  if (auto constant = getConstantIntValue(begin))
    return !extent && *constant >= 0 && *constant <= staticExtent && width <= staticExtent - *constant;
  auto coordinate = dyn_cast<BlockArgument>(begin);
  auto traversal = coordinate ? dyn_cast<scf::ForOp>(coordinate.getOwner()->getParentOp()) : scf::ForOp{};
  auto step = traversal ? getConstantIntValue(traversal.getStep()) : std::nullopt;
  if (!traversal || begin != traversal.getInductionVar() || !step || *step != width ||
      !matchPattern(traversal.getLowerBound(), m_Zero())) return false;
  Value end = traversal.getUpperBound();
  if (auto constant = getConstantIntValue(end)) {
    bool complete = !extent && *constant >= 0 && *constant <= staticExtent && *constant % width == 0;
    if (complete) vectorLimit = end;
    return complete;
  }
  // Full blocks use either n - n % width or (n / width) * width.
  // A plain vector.load does not itself promise that its entire slice is in bounds.
  if (auto subtract = end.getDefiningOp<arith::SubIOp>()) {
    auto remainder = subtract.getRhs().getDefiningOp<arith::RemSIOp>();
    bool complete = isExtent(subtract.getLhs()) && remainder && isExtent(remainder.getLhs()) &&
        getConstantIntValue(remainder.getRhs()) == width;
    if (complete) vectorLimit = end;
    return complete;
  }
  if (auto product = end.getDefiningOp<arith::MulIOp>()) {
    Value quotient;
    if (getConstantIntValue(product.getLhs()) == width) quotient = product.getRhs();
    else if (getConstantIntValue(product.getRhs()) == width) quotient = product.getLhs();
    auto division = quotient ? quotient.getDefiningOp<arith::DivSIOp>() : arith::DivSIOp{};
    bool complete = division && isExtent(division.getLhs()) && getConstantIntValue(division.getRhs()) == width;
    if (complete) vectorLimit = end;
    return complete;
  }
  return false;
}

bool fuse(memref::AllocOp allocation) {
  memref::StoreOp store;
  SmallVector<Operation *> loads;
  llvm::DenseMap<Operation *, Value> vectorLimits;
  SmallVector<memref::DeallocOp> deallocations;
  for (Operation *user : allocation->getUsers()) {
    if (auto write = dyn_cast<memref::StoreOp>(user)) {
      if (store) return false;
      store = write;
    } else if (auto read = dyn_cast<memref::LoadOp>(user)) {
      loads.push_back(read);
    } else if (auto read = dyn_cast<vector::LoadOp>(user)) {
      Value limit;
      if (!fullVectorRead(read, allocation, limit)) return false;
      vectorLimits[read] = limit;
      loads.push_back(read);
    } else if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      deallocations.push_back(dealloc);
    } else return false;
  }
  if (!store || loads.empty()) return false;
  SmallVector<scf::ForOp> loops;
  Operation *root = store;
  while (root->getBlock() != allocation->getBlock()) {
    auto parent = dyn_cast<scf::ForOp>(root->getParentOp());
    if (!parent || parent.getNumResults() || !matchPattern(parent.getLowerBound(), m_Zero()) ||
        !matchPattern(parent.getStep(), m_One())) return false;
    loops.push_back(parent);
    root = parent;
  }
  std::reverse(loops.begin(), loops.end());
  if (root->getBlock() != allocation->getBlock()) return false;
  SmallVector<AffineExpr> storedAxes;
  for (Value index : store.getIndices()) {
    auto loop = llvm::find_if(loops, [&](scf::ForOp loop) {
      return index == loop.getInductionVar();
    });
    if (loop != loops.end()) {
      storedAxes.push_back(getAffineDimExpr(loop - loops.begin(), allocation.getContext()));
    } else if (matchPattern(index, m_Zero())) {
      storedAxes.push_back(getAffineConstantExpr(0, allocation.getContext()));
    } else return false;
  }
  AffineMap outputMap = AffineMap::get(loops.size(), 0, storedAxes, allocation.getContext());
  auto projection = fullOutputProjection(allocation, outputMap);
  if (failed(projection)) return false;
  for (auto [axis, expression] : llvm::enumerate(storedAxes)) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension) continue;
    auto loop = loops[dimension.getPosition()];
    if (allocation.getType().isDynamicDim(axis)) {
      if (loop.getUpperBound() != allocation.getDynamicSizes()[
          allocation.getType().getDynamicDimIndex(axis)]) return false;
    } else {
      auto extent = getConstantIntValue(loop.getUpperBound());
      if (!extent || *extent != allocation.getType().getDimSize(axis)) return false;
    }
  }
  auto function = allocation->getParentOfType<func::FuncOp>();
  StorageAnalysis storage(function);
  // Erasing the traversal also erases operations outside the replayed value
  // slice. Their effects must be known reads, apart from this exact store.
  auto effects = storage.effects(root);
  if (!effects.complete || effects.ordered) return false;
  for (const StorageEffect &entry : effects.entries)
    if (entry.operation != store && !isa<MemoryEffects::Read>(entry.effect.getEffect()))
      return false;
  SmallVector<Value> frontier;
  for (auto loop : loops) frontier.push_back(loop.getInductionVar());
  auto payload = analyzeProducerValue(store.getValue(), root, frontier, storage);
  if (failed(payload)) return false;
  std::optional<ProducerVectorization> vectorization;
  if (llvm::any_of(loads, [](Operation *load) { return isa<vector::LoadOp>(load); })) {
    // SIMD advances the last memory axis, which need not be the last producer
    // loop after a permutation or the removal of unit output axes.
    if (auto dimension = dyn_cast<AffineDimExpr>(storedAxes.back())) {
      vectorization.emplace(*payload, loops[dimension.getPosition()].getInductionVar());
      if (!vectorization->canWiden(store.getValue())) return false;
    }
  }
  SmallVector<ProducerReplayUse> uses;
  for (Operation *load : loads) {
    ValueRange indices = isa<memref::LoadOp>(load) ? cast<memref::LoadOp>(load).getIndices()
                                                 : cast<vector::LoadOp>(load).getIndices();
    int64_t width = isa<vector::LoadOp>(load)
        ? cast<vector::LoadOp>(load).getVectorType().getDimSize(0) : 1;
    uses.push_back({load, llvm::to_vector(indices), width, vectorLimits.lookup(load)});
  }
  auto groups = groupProducerReplays(*payload, store.getValue(), root,
      allocation->getBlock(), uses, /*removesProducer=*/true, storage);
  if (failed(groups)) return false;
  for (const auto &group : *groups) {
    IRMapping mapping;
    for (auto [loop, expression] : llvm::zip(loops, projection->getResults()))
      mapping.map(loop.getInductionVar(), group.coordinates[
          cast<AffineDimExpr>(expression).getPosition()]);
    OpBuilder builder(group.anchor);
    Value replacement;
    if (isa<vector::LoadOp>(group.uses.front()) && vectorization) {
      IRMapping vectors;
      auto replayed = vectorization->materialize(store.getValue(),
          group.vectorWidth, builder, mapping, vectors);
      assert(succeeded(replayed) && "buffer replay must bind the proven vector payload");
      replacement = *replayed;
    } else {
      auto replayed = materializeProducerValue(*payload, store.getValue(), builder, mapping);
      assert(succeeded(replayed) && "buffer replay must bind the complete coordinate frontier");
      replacement = *replayed;
      if (isa<vector::LoadOp>(group.uses.front()))
        replacement = builder.create<vector::BroadcastOp>(group.anchor->getLoc(),
            group.uses.front()->getResult(0).getType(), replacement);
    }
    for (Operation *load : group.uses) {
      load->getResult(0).replaceAllUsesWith(replacement);
      load->erase();
    }
  }
  root->erase();
  for (memref::DeallocOp dealloc : deallocations) dealloc.erase();
  allocation.erase();
  return true;
}

}

LogicalResult fuseIntermediateBuffers(func::FuncOp function) {
  auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  if (!interface || !interface.getDisjointOutputs())
    return function.emitError("CPU buffer fusion requires established external alias legality");
  forwardDestinations(function);
  SmallVector<memref::CopyOp> copies;
  function.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    OpBuilder builder(copy);
    Location loc = copy.getLoc();
    SmallVector<Value> indices;
    std::function<void(int64_t)> materialize = [&](int64_t axis) {
      if (axis == copy.getSource().getType().getRank()) {
        Value value = builder.create<memref::LoadOp>(loc, copy.getSource(), indices);
        builder.create<memref::StoreOp>(loc, value, copy.getTarget(), indices);
        return;
      }
      Value extent = builder.create<memref::DimOp>(loc, copy.getSource(), axis);
      loop(builder, loc, index(builder, loc, 0), extent, 1, [&](Value coordinate) {
        indices.push_back(coordinate);
        materialize(axis + 1);
        indices.pop_back();
      });
    };
    materialize(0);
    copy.erase();
  }
  bool changed;
  do {
    changed = false;
    if (failed(foldPrivateAccessAliases(function))) return failure();
    SmallVector<memref::AllocOp> allocations;
    function.walk([&](memref::AllocOp op) { allocations.push_back(op); });
    for (memref::AllocOp allocation : llvm::reverse(allocations))
      changed |= fuse(allocation);
  } while (changed);
  return success();
}

void forwardCPUOutputs(func::FuncOp function) { forwardDestinations(function); }

}
