#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "ProducerReuse.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Analysis/ViewRelations.h"
#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool sameShape(Value lhs, Value rhs) {
  auto first = dyn_cast<MemRefType>(lhs.getType());
  auto second = dyn_cast<MemRefType>(rhs.getType());
  if (!first || !second || first.getRank() != second.getRank() ||
      first.getElementType() != second.getElementType() ||
      first.getMemorySpace() != second.getMemorySpace()) return false;
  for (int64_t axis = 0; axis < first.getRank(); ++axis)
    if (!haveEqualExtents(ValueBoundsConstraintSet::Variable(lhs, axis),
                          ValueBoundsConstraintSet::Variable(rhs, axis)))
      return false;
  return true;
}

bool completeWrite(Operation *operation, Value buffer) {
  if (auto store = dyn_cast<memref::StoreOp>(operation))
    return store.getMemref() == buffer && store.getMemRefType().getRank() == 0;
  if (auto copy = dyn_cast<memref::CopyOp>(operation))
    return copy.getTarget() == buffer && sameShape(copy.getSource(), buffer);
  auto writer = dyn_cast<linalg::LinalgOp>(operation);
  if (!writer || operation->getNumResults() || writer.getNumDpsInits() != 1 ||
      writer.getDpsInits().front() != buffer || writer.getNumReductionLoops() ||
      failed(fullOutputProjection(buffer, writer.getIndexingMapsArray().back())))
    return false;
  // A bijective map is not enough: the actual loop bounds must cover every
  // destination element, including dynamic dimensions and omitted unit axes.
  AffineMap bounds = writer.getShapesToLoopsMap();
  if (!bounds || bounds.getNumSymbols()) return false;
  SmallVector<std::pair<Value, int64_t>> dimensions;
  for (OpOperand &operand : operation->getOpOperands())
    for (int64_t axis = 0, rank = writer.getShape(&operand).size(); axis < rank; ++axis)
      dimensions.emplace_back(operand.get(), axis);
  for (auto [axis, expression] : llvm::enumerate(writer.getIndexingMapsArray().back().getResults())) {
    auto loop = dyn_cast<AffineDimExpr>(expression);
    if (!loop) continue;
    auto dimension = dyn_cast<AffineDimExpr>(bounds.getResult(loop.getPosition()));
    if (!dimension || dimension.getPosition() >= dimensions.size()) return false;
    auto [value, sourceAxis] = dimensions[dimension.getPosition()];
    if (!haveEqualExtents(ValueBoundsConstraintSet::Variable(buffer, axis),
                          ValueBoundsConstraintSet::Variable(value, sourceAxis)))
      return false;
  }
  return true;
}

bool disjointEffect(Value memory, Value buffer, StorageAnalysis &storage) {
  return memory && (!isa<BaseMemRefType>(memory.getType()) ||
                    storage.disjoint(memory, buffer));
}

bool knownVersionEffects(Operation *operation, Value buffer,
                         const StorageEffects &effects, StorageAnalysis &storage) {
  if (!effects.complete) return false;
  if (!effects.ordered) return true;
  Value origin = storage.uniqueOrigin(buffer);
  if (!origin || !isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(origin.getDefiningOp()))
    return false;
  // This observes private contents; it does not move a memory access. Ordering
  // on completely independent addressable storage cannot change this version.
  // Replay and read placement still use the stronger readStable contract.
  for (const StorageEffect &entry : effects.entries) {
    Value memory = entry.effect.getValue();
    if (!memory || !isa<BaseMemRefType>(memory.getType()) ||
        !storage.disjoint(memory, buffer)) return false;
  }
  // An unaddressed ordering barrier is not explained by disjoint effects on
  // another operation in the same enclosing region.
  auto result = operation->walk([&](Operation *nested) {
    auto current = storage.effects(nested);
    return current.ordered && current.entries.empty()
        ? WalkResult::interrupt() : WalkResult::advance();
  });
  return !result.wasInterrupted();
}

std::optional<unsigned> readingInput(Operation *operation, Value buffer,
                                     StorageAnalysis &storage) {
  if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
    if (copy.getSource() == buffer && sameShape(buffer, copy.getTarget()) &&
        storage.disjoint(buffer, copy.getTarget())) return 0;
    return std::nullopt;
  }
  auto generic = dyn_cast<linalg::GenericOp>(operation);
  auto reduce = dyn_cast<ReduceOp>(operation);
  if ((!generic && !reduce) || (generic &&
      (generic.getNumResults() || generic.getOutputs().size() != 1)))
    return std::nullopt;
  ValueRange inputs = generic ? ValueRange(generic.getInputs())
                             : ValueRange(reduce.getInputs());
  SmallVector<AffineMap> maps = generic ? generic.getIndexingMapsArray()
      : llvm::to_vector(llvm::map_range(reduce.getIndexingMaps(), [](Attribute attr) {
          return cast<AffineMapAttr>(attr).getValue();
        }));
  std::optional<unsigned> selected;
  for (auto [number, input] : llvm::enumerate(inputs)) {
    if (!isa<BaseMemRefType>(input.getType()) || storage.disjoint(input, buffer)) continue;
    if (input != buffer || (selected && maps[*selected] != maps[number]))
      return std::nullopt;
    selected = number;
  }
  if (!selected) return std::nullopt;
  Block &body = generic ? generic.getRegion().front() : reduce.getCombine().front();
  if (generic && !storage.disjoint(generic.getOutputs()[0], buffer) &&
      !body.getArguments().back().use_empty()) return std::nullopt;
  // Implicit input slots above are the complete observation. Explicit reads or
  // writes through another descriptor cannot be transported by that mapping.
  for (Operation &nested : body.without_terminator()) {
    auto effects = storage.effects(&nested);
    if (!effects.complete || effects.ordered) return std::nullopt;
    for (const StorageEffect &entry : effects.entries)
      if (!disjointEffect(entry.effect.getValue(), buffer, storage))
        return std::nullopt;
  }
  return selected;
}

bool endsAtIteration(Value origin, Operation *completed,
                     ArrayRef<Operation *> reads, StorageAnalysis &storage) {
  auto loop = dyn_cast<scf::ForOp>(completed->getParentOp());
  auto allocation = origin.getDefiningOp<memref::AllocOp>();
  if (!loop || completed->getBlock() != loop.getBody() || !allocation ||
      !completeWrite(completed, origin)) return false;
  auto lifetime = storage.lifetime(allocation);
  if (!lifetime || !lifetime->contains(loop)) return false;

  // A lexical writer is not a unique dynamic writer across tasks or parallel
  // regions. Only sequential control may separate this allocation instance
  // from the loop that reuses it.
  for (Operation *scope = loop; scope->getBlock() != allocation->getBlock();) {
    scope = scope->getParentOp();
    if (!scope || !isa<scf::ForOp, scf::IfOp>(scope)) return false;
  }
  DominanceInfo dominance(completed->getParentOfType<func::FuncOp>());
  for (Operation *user : lifetime->aliases.users) {
    if (user == lifetime->end) continue;
    auto effects = storage.effects(user);
    if (!knownVersionEffects(user, origin, effects, storage)) return false;
    if ((isStorageAliasOperation(user) || isa<memref::DimOp, memref::RankOp>(user)) &&
        effects.entries.empty() && !effects.ordered) continue;
    bool observes = false;
    for (const StorageEffect &entry : effects.entries) {
      if (disjointEffect(entry.effect.getValue(), origin, storage)) continue;
      observes = true;
      if (isa<MemoryEffects::Write>(entry.effect.getEffect())) {
        if (entry.operation != completed) return false;
        continue;
      }
      if (!isa<MemoryEffects::Read>(entry.effect.getEffect()) ||
          completed->isAncestor(entry.operation) ||
          !dominance.properlyDominates(completed, entry.operation) ||
          !llvm::is_contained(reads, entry.operation)) return false;
      // Nested counted reads stay in this iteration. Crossing another execution
      // or conditional boundary needs a stronger observation contract.
      Operation *scope = entry.operation;
      while (scope->getBlock() != loop.getBody()) {
        scope = scope->getParentOp();
        if (!scope || !isa<scf::ForOp>(scope)) return false;
      }
    }
    if (!observes) return false;
  }
  return true;
}

} // namespace

std::optional<CurrentBufferWrite> findCurrentBufferWrite(
    Value buffer, Operation *before, StorageAnalysis &storage) {
  Value origin = storage.uniqueOrigin(buffer);
  if (!origin || !storage.aliases(origin).complete)
    return std::nullopt;
  auto function = before->getParentOfType<func::FuncOp>();
  DominanceInfo dominance(function);
  if (!dominance.dominates(buffer, before)) return std::nullopt;
  if (auto allocation = origin.getDefiningOp<memref::AllocOp>()) {
    auto lifetime = storage.lifetime(allocation);
    if (!lifetime || !lifetime->contains(before)) return std::nullopt;
  } else if (auto allocation = origin.getDefiningOp<memref::AllocaOp>()) {
    auto owner = allocation->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
    if (!owner || !owner->isAncestor(before)) return std::nullopt;
  } else {
    auto formal = dyn_cast<BlockArgument>(origin);
    if (!formal || (formal.getOwner() != &function.front() &&
        (!isCollectiveArgument(formal) ||
         !formal.getOwner()->getParent()->isAncestor(before->getParentRegion()))))
      return std::nullopt;
  }
  bool observed = false;
  for (Operation *point = before; point && !isa<func::FuncOp>(point);) {
    for (Operation *operation = point->getPrevNode(); operation;
         operation = operation->getPrevNode()) {
      auto effects = storage.effects(operation);
      if (!knownVersionEffects(operation, buffer, effects, storage)) return std::nullopt;
      bool written = false;
      for (const StorageEffect &entry : effects.entries) {
        const auto &effect = entry.effect;
        if (disjointEffect(effect.getValue(), buffer, storage)) continue;
        if (isa<MemoryEffects::Read>(effect.getEffect())) {
          observed = true;
          continue;
        }
        if (!isa<MemoryEffects::Write>(effect.getEffect())) return std::nullopt;
        written = true;
      }
      if (written) return CurrentBufferWrite{operation, observed};
    }
    Operation *owner = point->getParentOp();
    if (!owner || isa<func::FuncOp>(owner)) return std::nullopt;
    auto ownerEffects = storage.effects(owner);
    if (!knownVersionEffects(owner, buffer, ownerEffects, storage) ||
        !storage.preservesContents(owner, buffer)) return std::nullopt;
    for (const StorageEffect &entry : ownerEffects.entries)
      if (isa<MemoryEffects::Read>(entry.effect.getEffect()) &&
          !disjointEffect(entry.effect.getValue(), buffer, storage)) observed = true;
    point = owner;
  }
  return std::nullopt;
}

bool completelyWritesBuffer(Operation *operation, Value buffer) {
  return completeWrite(operation, buffer);
}

std::optional<UniformBufferValue> queryUniformBufferValue(
    Value buffer, Operation *before, StorageAnalysis &storage) {
  auto current = findCurrentBufferWrite(buffer, before, storage);
  if (!current || !completeWrite(current->operation, buffer)) return std::nullopt;
  Value value;
  if (auto fill = dyn_cast<linalg::FillOp>(current->operation))
    value = fill.getInputs().front();
  else if (auto store = dyn_cast<memref::StoreOp>(current->operation))
    value = store.getValue();
  else if (auto copy = dyn_cast<memref::CopyOp>(current->operation)) {
    auto input = queryUniformBufferValue(copy.getSource(), copy, storage);
    if (input) value = input->value;
  }
  if (!value) return std::nullopt;
  return UniformBufferValue{*current, value};
}

void eraseReplayedProducerVersion(linalg::GenericOp producer) {
  Value buffer = producer.getOutputs().front();
  if (buffer.getDefiningOp<memref::AllocOp>() &&
      llvm::all_of(buffer.getUsers(), [&](Operation *user) {
        return user == producer || isRemovableProducerMetadata(user);
      }))
    eraseUnusedProducer(producer);
  else
    producer.erase();
}

FailureOr<ProducerVersion> queryProducerVersion(linalg::GenericOp producer,
                                               StorageAnalysis &storage) {
  if (producer.getOutputs().size() != 1) return failure();
  Value buffer = producer.getOutputs()[0];
  if (!completeWrite(producer, buffer) || !isContiguousDescriptor(buffer) ||
      !producer.getRegion().front().getArguments().back().use_empty())
    return failure();
  Value origin = storage.uniqueOrigin(buffer);
  if (!origin || !storage.aliases(origin).complete) return failure();
  auto produced = storage.effects(producer);
  if (!produced.complete || produced.ordered) return failure();
  for (const StorageEffect &entry : produced.entries)
    if (isa<MemoryEffects::Read>(entry.effect.getEffect()) &&
        !disjointEffect(entry.effect.getValue(), buffer, storage))
      return failure();

  return queryCompletedBufferVersion(buffer, producer, storage);
}

FailureOr<ProducerVersion> queryCompletedBufferVersion(
    Value buffer, Operation *completed, StorageAnalysis &storage) {
  auto observations = queryBufferVersionObservations(buffer, completed, storage);
  if (failed(observations)) return failure();
  ProducerVersion result;
  for (Operation *read : observations->reads) {
    if (read->getBlock() != completed->getBlock()) return failure();
    auto input = readingInput(read, buffer, storage);
    if (!input) return failure();
    result.uses.push_back({read, *input});
  }
  return result;
}

FailureOr<BufferVersionObservations> queryBufferVersionObservations(
    Value buffer, Operation *completed, StorageAnalysis &storage) {
  Value origin = storage.uniqueOrigin(buffer);
  if (!origin || !storage.aliases(origin).complete) return failure();
  BufferVersionObservations result;
  for (Operation *operation = completed->getNextNode(); operation;
       operation = operation->getNextNode()) {
    auto effects = storage.effects(operation);
    if (!knownVersionEffects(operation, buffer, effects, storage)) return failure();
    bool writes = false, frees = false;
    for (const StorageEffect &entry : effects.entries) {
      const auto &effect = entry.effect;
      if (disjointEffect(effect.getValue(), buffer, storage)) continue;
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        if (!llvm::is_contained(result.reads, entry.operation))
          result.reads.push_back(entry.operation);
      } else if (isa<MemoryEffects::Write>(effect.getEffect())) writes = true;
      else if (isa<MemoryEffects::Free>(effect.getEffect())) frees = true;
      else return failure();
    }
    if (writes) {
      if (!completeWrite(operation, buffer)) return failure();
      result.end = operation;
      return result;
    }
    if (frees) {
      auto release = dyn_cast<memref::DeallocOp>(operation);
      if (!release || release.getMemref() != buffer ||
          !buffer.getDefiningOp<memref::AllocOp>()) return failure();
      result.end = operation;
      return result;
    }
  }
  // The next iteration starts with the same complete definition. On exit no
  // contents observer remains, including through aliases outside the loop.
  if (endsAtIteration(origin, completed, result.reads, storage)) {
    result.end = completed->getBlock()->getTerminator();
    return result;
  }
  // A surviving output is an observation too. A lexical block end alone does
  // not establish that this version can be deleted.
  return failure();
}

bool forwardProducerVersionCopies(func::FuncOp function) {
  SmallVector<linalg::GenericOp> producers;
  function.walk([&](linalg::GenericOp producer) {
    if (!producer.getNumResults() && producer.getOutputs().size() == 1 &&
        !producer.getNumReductionLoops()) producers.push_back(producer);
  });
  StorageAnalysis storage(function);
  DominanceInfo dominance(function);
  for (auto producer : producers) {
    auto version = queryProducerVersion(producer, storage);
    if (failed(version) || version->uses.empty() ||
        failed(analyzeLinalgProducer(producer, storage))) continue;
    Value source = producer.getOutputs()[0];
    for (const ProducerVersionUse &use : version->uses) {
      auto copy = dyn_cast<memref::CopyOp>(use.operation);
      if (!copy) continue;
      Value target = copy.getTarget();
      if (!isContiguousDescriptor(target)) continue;
      Value origin = storage.uniqueOrigin(target);
      if (!origin || !storage.aliases(origin).complete) continue;

      Operation *descriptor = nullptr;
      if (!dominance.properlyDominates(target, producer)) {
        descriptor = target.getDefiningOp();
        if (!descriptor || descriptor->getBlock() != producer->getBlock() ||
            !isStorageAliasOperation(descriptor) || descriptor->getNumRegions() ||
            !isMemoryEffectFree(descriptor) || !isSpeculatable(descriptor) ||
            llvm::any_of(descriptor->getOperands(), [&](Value operand) {
              return !dominance.properlyDominates(operand, producer);
            })) continue;
      }

      // Moving the destination write from the copy back to the producer must
      // not change any observation of its previous contents, including reads
      // in the producer itself. The source and target were proved disjoint by
      // the copy observation query, so the original defining write is harmless.
      bool unobserved = true;
      for (Operation *operation = producer; operation != copy;
           operation = operation->getNextNode()) {
        auto effects = storage.effects(operation);
        if (!effects.complete || effects.ordered) { unobserved = false; break; }
        for (const StorageEffect &entry : effects.entries)
          if (!disjointEffect(entry.effect.getValue(), target, storage)) {
            unobserved = false;
            break;
          }
        if (!unobserved) break;
      }
      if (!unobserved) continue;

      // Existing consumers before the copy will see the newly retargeted write.
      // After it, the copied snapshot must remain stable through the last
      // redirected read. Later writes to either destination retain their scope.
      Operation *last = version->uses.back().operation;
      if (last != copy &&
          (!storage.unchangedBetween(target, copy, last) ||
           !storage.preserves(last, target))) continue;

      SmallVector<OpOperand *> reads;
      for (const ProducerVersionUse &reader : version->uses) {
        if (reader.operation == copy) continue;
        if (auto generic = dyn_cast<linalg::GenericOp>(reader.operation)) {
          for (OpOperand *operand : generic.getDpsInputOperands())
            if (operand->get() == source) reads.push_back(operand);
        } else if (auto transfer = dyn_cast<memref::CopyOp>(reader.operation)) {
          reads.push_back(&transfer->getOpOperand(0));
        } else {
          assert(isa<ReduceOp>(reader.operation));
          for (OpOperand &operand : reader.operation->getOpOperands())
            if (operand.get() == source) reads.push_back(&operand);
        }
      }
      if (descriptor) descriptor->moveBefore(producer);
      producer.getDpsInitsMutable()[0].set(target);
      for (OpOperand *read : reads) read->set(target);
      copy.erase();
      return true; // Rebuild the snapshot before considering another version.
    }
  }
  return false;
}

bool canReplayProducerVersionAt(const ProducerReplay &payload,
    linalg::GenericOp producer, Operation *consumer, AffineMap coordinates,
    StorageAnalysis &storage) {
  auto generic = dyn_cast<linalg::GenericOp>(consumer);
  ValueRange outputs = generic ? ValueRange(generic.getOutputs()) : ValueRange{};
  if (canReplayProducerAt(payload, producer, consumer, storage, outputs)) return true;
  if (!generic || generic.getNumResults() || generic.getOutputs().size() != 1 ||
      generic.getNumReductionLoops() ||
      !generic.getRegion().front().getArguments().back().use_empty()) return false;
  Value output = generic.getOutputs()[0];
  if (!completeWrite(generic, output) || !isContiguousDescriptor(output)) return false;
  for (auto operation : {producer, generic})
    for (Operation &nested : operation.getRegion().front().without_terminator())
      if (nested.getNumRegions() || !isMemoryEffectFree(&nested)) return false;
  AffineMap outputMap = generic.getIndexingMapsArray().back();
  bool sourceFound = false;
  for (auto [input, map] : llvm::zip(producer.getInputs(), producer.getIndexingMapsArray())) {
    if (!isa<BaseMemRefType>(input.getType()) || storage.disjoint(input, output)) continue;
    if (input != output || map.compose(coordinates) != outputMap) return false;
    sourceFound = true;
  }
  if (!sourceFound || !storage.preserves(producer, output) ||
      !storage.unchangedBetween(output, producer, consumer)) return false;
  ProducerReplay remaining = payload;
  llvm::erase_if(remaining.reads, [&](Value memory) { return memory == output; });
  // Only the exact, injective endpoint read above has a different movement
  // proof. Captures, lifetime and every other read retain the common contract.
  return canReplayProducerAt(remaining, producer, consumer, storage, outputs);
}

} // namespace intent::cpu
