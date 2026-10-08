#include "Intent/Dialect/CPU/Transforms/Control/Traversals.h"
#include "TraversalFusion.h"
#include "../Storage/AccessAliases.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

using detail::Traversal;

std::optional<Traversal> traversal(Operation *operation) {
  if (auto loop = dyn_cast<scf::ForOp>(operation))
    return Traversal{loop, loop.getBody(), {loop.getLowerBound()}, {loop.getUpperBound()},
                     {loop.getStep()}, {loop.getInductionVar()}};
  if (auto loop = dyn_cast<scf::ParallelOp>(operation); loop && !loop.getNumResults())
    return Traversal{loop, loop.getBody(), llvm::to_vector(loop.getLowerBound()),
                     llvm::to_vector(loop.getUpperBound()), llvm::to_vector(loop.getStep()),
                     llvm::to_vector(loop.getInductionVars())};
  return std::nullopt;
}

bool sameBounds(const Traversal &first, const Traversal &second) {
  if (first.operation->getName() != second.operation->getName() ||
      first.coordinates.size() != second.coordinates.size()) return false;
  auto equal = [](Value lhs, Value rhs) {
    if (lhs == rhs) return true;
    auto left = getConstantIntValue(lhs), right = getConstantIntValue(rhs);
    if (left && right) return *left == *right;
    return haveEqualExtents(ValueBoundsConstraintSet::Variable(lhs),
                            ValueBoundsConstraintSet::Variable(rhs));
  };
  return llvm::all_of(llvm::zip(first.lower, second.lower, first.upper, second.upper,
                                first.step, second.step), [&](auto bounds) {
    return equal(std::get<0>(bounds), std::get<1>(bounds)) &&
           equal(std::get<2>(bounds), std::get<3>(bounds)) &&
           equal(std::get<4>(bounds), std::get<5>(bounds));
  });
}

bool compatibleAttributes(Operation *first, Operation *second,
                           ReductionOrderAttr &order) {
  constexpr StringLiteral orderName = "intent_cpu.reduction_order";
  for (Operation *source : {first, second}) {
    Operation *other = source == first ? second : first;
    for (NamedAttribute attribute : source->getAttrs())
      if (attribute.getName() != orderName &&
          other->getAttr(attribute.getName()) != attribute.getValue()) return false;
  }
  auto lhs = first->getAttrOfType<ReductionOrderAttr>(orderName);
  auto rhs = second->getAttrOfType<ReductionOrderAttr>(orderName);
  if (first->getNumResults() && second->getNumResults()) {
    // Ordinary carries do not inherit a reduction's permissions. Also retain
    // the SIMD freedom of either traversal instead of weakening it by fusion.
    if (lhs != rhs) return false;
    order = lhs;
  } else {
    order = first->getNumResults() ? lhs : second->getNumResults() ? rhs : ReductionOrderAttr{};
  }
  return true;
}

bool movableBetween(Operation *operation) {
  return isa<memref::AllocOp, memref::AllocaOp, memref::DeallocOp, memref::StoreOp>(operation) ||
      (!operation->getNumRegions() && isMemoryEffectFree(operation) && isSpeculatable(operation));
}

bool canDeferStore(memref::StoreOp store, Operation *across, StorageAnalysis &storage) {
  auto effects = storage.effects(across);
  if (!effects.complete || effects.ordered) return false;
  for (const StorageEffect &entry : effects.entries) {
    Value memory = entry.effect.getValue();
    if (!memory || !isa<MemRefType>(memory.getType()) ||
        !storage.disjoint(store.getMemref(), memory)) return false;
  }
  return true;
}

bool canFuse(const Traversal &first, const Traversal &second,
             ArrayRef<Operation *> between, StorageAnalysis &storage,
             DominanceInfo &dominance,
             SmallVectorImpl<Operation *> &deferred) {
  if (!sameBounds(first, second)) return false;
  llvm::SmallPtrSet<Operation *, 16> movable;
  for (Operation *operation : between) {
    if (!movableBetween(operation)) return false;
    if (isa<memref::DeallocOp>(operation)) continue;
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      if (!canDeferStore(store, second.operation, storage)) return false;
      deferred.push_back(operation);
      continue;
    }
    bool availableBefore = llvm::all_of(operation->getOperands(), [&](Value operand) {
      return dominance.properlyDominates(operand, first.operation) ||
             movable.contains(operand.getDefiningOp());
    });
    if (!availableBefore) {
      if (isa<memref::AllocOp, memref::AllocaOp>(operation)) return false;
      deferred.push_back(operation);
      continue;
    }
    movable.insert(operation);
  }
  // Completed scalar postprocessing and disjoint output stores may remain
  // after both traversals, without supplying either an earlier SSA observer
  // or any memory access in the second traversal.
  for (Operation *operation : deferred)
    for (Operation *user : operation->getUsers())
      if (!llvm::is_contained(deferred, user) &&
          (second.operation->isAncestor(user) ||
           !dominance.properlyDominates(second.operation, user)))
        return false;
  llvm::SetVector<Value> captures;
  captures.insert(second.operation->operand_begin(), second.operation->operand_end());
  for (Region &region : second.operation->getRegions())
    getUsedValuesDefinedAbove(region, captures);
  for (Value value : captures)
    if (!dominance.properlyDominates(value, first.operation) &&
        !movable.contains(value.getDefiningOp())) return false;
  // Native fusion inserts after the second traversal. The only uses that may
  // precede it are the proved postprocessing DAG moved to that same final scope.
  for (Operation *user : first.operation->getUsers())
    if (!llvm::is_contained(deferred, user) &&
        (second.operation->isAncestor(user) ||
         !dominance.properlyDominates(second.operation, user))) return false;

  auto lhs = detail::traversalAccesses(first, storage);
  auto rhs = detail::traversalAccesses(second, storage);
  if (failed(lhs) || failed(rhs)) return false;
  IRMapping mapping;
  mapping.map(first.coordinates, second.coordinates);
  bool shared = false;
  for (const auto &left : *lhs) {
    for (const auto &right : *rhs) {
      bool same = detail::sameTraversalAddress(left, right, mapping);
      // Sharing needs an actual read footprint or producer-consumer relation;
      // merely using the same allocation is not a benefit or legality proof.
      if (same && (!left.write || !right.write)) shared = true;
      if (!shared && !left.write && !right.write)
        shared = detail::sameNestedReadFootprint(left, right, first, second, mapping);
      if (!left.write && !right.write) continue;
      if (storage.disjoint(left.memory, right.memory)) continue;
      if (!same || !detail::separatesIterations(left, first, dominance) ||
          !detail::separatesIterations(right, second, dominance)) return false;
    }
  }
  return shared;
}

void join(const Traversal &first, const Traversal &second,
          ArrayRef<Operation *> between, ArrayRef<Operation *> deferred,
          ReductionOrderAttr order) {
  Operation *freePoint = second.operation;
  for (Operation *operation : deferred) {
    operation->moveAfter(freePoint);
    freePoint = operation;
  }
  for (Operation *operation : between) {
    if (isa<memref::DeallocOp>(operation)) {
      operation->moveAfter(freePoint);
      freePoint = operation;
    } else if (!llvm::is_contained(deferred, operation))
      operation->moveBefore(first.operation);
  }
  NamedAttrList attributes(first.operation->getAttrs());
  attributes.erase("intent_cpu.reduction_order");
  if (order) attributes.set("intent_cpu.reduction_order", order);
  IRRewriter rewriter(first.operation->getContext());
  if (auto a = dyn_cast<scf::ForOp>(first.operation)) {
    auto combined = fuseIndependentSiblingForLoops(
        a, cast<scf::ForOp>(second.operation), rewriter);
    combined->setAttrs(attributes);
  } else {
    // MLIR's public parallel fusion driver also owns a weaker, load/store-only
    // legality policy. Use its block-rewrite primitives after our current-IR
    // proof, preserving nested regions and the original parallel operation.
    rewriter.eraseOp(second.body->getTerminator());
    rewriter.inlineBlockBefore(second.body, first.body->getTerminator(), first.coordinates);
    rewriter.eraseOp(second.operation);
    first.operation->setAttrs(attributes);
  }
}

bool fuseOne(func::FuncOp function) {
  SmallVector<Operation *> candidates;
  function.walk([&](Operation *operation) {
    if (traversal(operation)) candidates.push_back(operation);
  });
  StorageAnalysis storage(function);
  DominanceInfo dominance(function);
  for (Operation *operation : candidates) {
    auto first = *traversal(operation);
    SmallVector<Operation *> between;
    Operation *next = operation->getNextNode();
    while (next && !traversal(next) && movableBetween(next)) {
      between.push_back(next);
      next = next->getNextNode();
    }
    auto second = next ? traversal(next) : std::nullopt;
    if (!second) continue;
    ReductionOrderAttr order;
    SmallVector<Operation *> deferred;
    if (!compatibleAttributes(first.operation, second->operation, order) ||
        !canFuse(first, *second, between, storage, dominance, deferred)) continue;
    join(first, *second, between, deferred, order);
    return true;
  }
  return false;
}

} // namespace

LogicalResult fuseSharedTraversals(func::FuncOp function) {
  bool changed;
  do {
    if (failed(foldPrivateAccessAliases(function))) return failure();
    changed = fuseOne(function);
    changed |= reuseMemoryValues(function);
    if (changed) {
      IRRewriter rewriter(function.getContext());
      DominanceInfo dominance(function);
      eliminateCommonSubExpressions(rewriter, dominance, function);
      if (failed(applyPatternsGreedily(function, RewritePatternSet(function.getContext()))))
        return failure();
      eraseDeadPrivateBuffers(function);
    }
  } while (changed);
  return success();
}

} // namespace intent::cpu
