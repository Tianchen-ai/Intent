#include "TraversalFusion.h"
#include "../Storage/MemoryAccess.h"
#include "Intent/Dialect/CPU/Analysis/ViewRelations.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace intent::cpu::detail {

bool sameTraversalAddress(const TraversalAccess &first, const TraversalAccess &second,
                          const IRMapping &mapping) {
  if (first.indexed != second.indexed || first.indices.size() != second.indices.size() ||
      !sameMemoryValue(first.memory, second.memory, mapping)) return false;
  if (!first.indexed && (!first.indexingMap || !second.indexingMap ||
                         first.indexingMap != second.indexingMap)) return false;
  return llvm::all_of(llvm::zip(first.indices, second.indices), [&](auto pair) {
    return sameMemoryValue(std::get<0>(pair), std::get<1>(pair), mapping);
  });
}

bool sameNestedReadFootprint(const TraversalAccess &first, const TraversalAccess &second,
                             const Traversal &firstTraversal,
                             const Traversal &secondTraversal,
                             const IRMapping &mapping) {
  if (first.write || second.write) return false;
  auto nest = [](Operation *access, Operation *root,
                 SmallVectorImpl<scf::ForOp> &loops) {
    for (Operation *scope = access->getParentOp(); scope != root;
         scope = scope->getParentOp()) {
      auto loop = dyn_cast_or_null<scf::ForOp>(scope);
      if (!loop) return false;
      loops.push_back(loop);
    }
    return true;
  };
  SmallVector<scf::ForOp> left, right;
  if (!nest(first.operation, firstTraversal.operation, left) ||
      !nest(second.operation, secondTraversal.operation, right) ||
      left.size() != right.size()) return false;
  IRMapping members(mapping);
  for (auto [a, b] : llvm::zip(llvm::reverse(left), llvm::reverse(right))) {
    if (a->getAttrs() != b->getAttrs() ||
        !sameMemoryValue(a.getLowerBound(), b.getLowerBound(), members) ||
        !sameMemoryValue(a.getUpperBound(), b.getUpperBound(), members) ||
        !sameMemoryValue(a.getStep(), b.getStep(), members)) return false;
    members.map(a.getInductionVar(), b.getInductionVar());
  }
  // Equal complete read domains are a reuse opportunity. This correspondence
  // deliberately does not participate in the write-dependence proof.
  return sameTraversalAddress(first, second, members);
}

FailureOr<SmallVector<TraversalAccess>> traversalAccesses(
    const Traversal &traversal, StorageAnalysis &storage) {
  auto effects = storage.effects(traversal.operation);
  if (!effects.complete || effects.ordered) return failure();
  SmallVector<TraversalAccess> accesses;
  for (const StorageEffect &entry : effects.entries) {
    const auto &effect = entry.effect;
    if (isa<MemoryEffects::Allocate>(effect.getEffect())) continue;
    Value memory = effect.getValue();
    if (!memory || !isa<MemRefType>(memory.getType())) return failure();
    auto origins = storage.origins(memory);
    bool local = origins.complete && !origins.values.empty() &&
        llvm::all_of(origins.values, [&](Value origin) {
          Operation *owner = origin.getDefiningOp();
          if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(owner) ||
              !traversal.operation->isAncestor(owner)) return false;
          auto aliases = storage.aliases(origin);
          return aliases.complete && llvm::all_of(aliases.users, [&](Operation *user) {
            return traversal.operation->isAncestor(user);
          });
        });
    if (local) continue;
    if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) return failure();
    TraversalAccess access{entry.operation, memory, {}, false,
                            isa<MemoryEffects::Write>(effect.getEffect())};
    if (auto load = dyn_cast<memref::LoadOp>(entry.operation)) {
      access.indices.assign(load.getIndices().begin(), load.getIndices().end());
      access.indexed = true;
    } else if (auto store = dyn_cast<memref::StoreOp>(entry.operation)) {
      access.indices.assign(store.getIndices().begin(), store.getIndices().end());
      access.indexed = true;
    } else if (auto generic = dyn_cast<linalg::LinalgOp>(entry.operation)) {
      for (OpOperand &operand : generic->getOpOperands()) {
        if (operand.get() != memory ||
            (access.write ? !generic.isDpsInit(&operand)
                          : !generic.payloadUsesValueFromOperand(&operand))) continue;
        access.indexingMap = generic.getMatchingIndexingMap(&operand);
        accesses.push_back(access);
      }
      continue;
    }
    accesses.push_back(std::move(access));
  }
  return accesses;
}

bool containedSubview(memref::SubViewOp view) {
  using Bounds = ValueBoundsConstraintSet;
  using Variable = Bounds::Variable;
  MLIRContext *context = view.getContext();
  auto zero = IntegerAttr::get(IndexType::get(context), 0);
  if (!llvm::all_of(view.getMixedStrides(), [](OpFoldResult stride) {
        return getConstantIntValue(stride) == 1;
      })) return false;
  AffineMap addition = AffineMap::get(2, 0,
      getAffineDimExpr(0, context) + getAffineDimExpr(1, context));
  auto offsets = view.getMixedOffsets(), sizes = view.getMixedSizes();
  for (auto [axis, pair] : llvm::enumerate(llvm::zip(offsets, sizes))) {
    Variable offset(std::get<0>(pair)), size(std::get<1>(pair));
    Variable end(addition, ArrayRef<Variable>{offset, size});
    if (!Bounds::compare(offset, Bounds::GE, Variable(zero)) ||
        !Bounds::compare(size, Bounds::GE, Variable(zero)) ||
        !Bounds::compare(end, Bounds::LE, Variable(view.getSource(), axis)))
      return false;
  }
  return true;
}

bool separatesIterations(const TraversalAccess &access, const Traversal &traversal,
                         DominanceInfo &dominance) {
  // A descriptor outside the traversal is fixed for all of its iterations.
  // Contiguity plus explicit distinct coordinates establishes disjoint cells.
  if (access.indexed && dominance.dominates(access.memory, traversal.operation) &&
      isContiguousDescriptor(access.memory) &&
      llvm::all_of(traversal.coordinates, [&](Value coordinate) {
        return llvm::is_contained(access.indices, coordinate);
      })) return true;
  Value memory = access.memory;
  while (Operation *owner = memory.getDefiningOp()) {
    if (auto cast = dyn_cast<memref::CastOp>(owner)) {
      memory = cast.getSource();
      continue;
    }
    auto view = dyn_cast<memref::SubViewOp>(owner);
    if (!view || !containedSubview(view)) return false;
    if (dominance.dominates(view.getSource(), traversal.operation) &&
        isContiguousDescriptor(view.getSource())) {
      auto offsets = view.getMixedOffsets(), sizes = view.getMixedSizes();
      return llvm::all_of(traversal.coordinates, [&](Value coordinate) {
        for (auto [offset, size] : llvm::zip(offsets, sizes))
          if (offset == OpFoldResult(coordinate) && getConstantIntValue(size) == 1)
            return true;
        return false;
      });
    }
    memory = view.getSource();
  }
  return false;
}

} // namespace intent::cpu::detail
