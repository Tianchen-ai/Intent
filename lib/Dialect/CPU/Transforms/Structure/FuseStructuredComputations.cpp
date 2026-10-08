#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/Computations.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerReplay.h"
#include "IntegerSources.h"
#include "ProducerReuse.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "ContiguousAccesses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {

namespace {

struct PointwiseProducer {
  linalg::GenericOp operation;
  ProducerReplay payload;
  AffineMap outputProjection;
};

bool fusionConsumer(Operation *operation, StorageAnalysis &storage) {
  if (isa<ReduceOp>(operation)) return true;
  auto generic = dyn_cast<linalg::GenericOp>(operation);
  if (!generic || isMatrixContraction(generic) || generic.getNumResults() ||
      generic.getOutputs().size() != 1 ||
      failed(analyzeLinalgProducer(generic, storage))) return false;
  if (generic.getNumReductionLoops()) {
    llvm::SmallBitVector outputAxes(generic.getNumLoops());
    for (AffineExpr expression : generic.getIndexingMapsArray().back().getResults()) {
      auto axis = dyn_cast<AffineDimExpr>(expression);
      if (!axis || outputAxes.test(axis.getPosition())) return false;
      outputAxes.set(axis.getPosition());
    }
    for (auto [axis, kind] : llvm::enumerate(generic.getIteratorTypesArray()))
      if (outputAxes.test(axis) != (kind == utils::IteratorType::parallel)) return false;
  }
  return true;
}

std::optional<PointwiseProducer> pointwiseProducer(Value buffer, Operation *consumer,
                                                  StorageAnalysis &storage) {
  if (!buffer.getDefiningOp<memref::AllocOp>()) return std::nullopt;
  auto current = findCurrentBufferWrite(buffer, consumer, storage);
  auto producer = current ? dyn_cast<linalg::GenericOp>(current->operation) : linalg::GenericOp{};
  if (!producer || producer == consumer || producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer) || producer.getOutputs().size() != 1 ||
      producer.getNumResults() || producer.getNumReductionLoops() ||
      !completelyWritesBuffer(producer, buffer) ||
      !storage.preserves(consumer, buffer)) return std::nullopt;
  auto projection = fullOutputProjection(buffer, producer.getIndexingMapsArray().back());
  if (failed(projection)) return std::nullopt;
  Block &body = producer.getRegion().front();
  if (!body.getArguments().back().use_empty()) return std::nullopt;
  // CPU Reduce's scalar combine has a deliberately narrower contract than a
  // linalg payload. Do not import regions or explicit reads into that ABI.
  if (isa<ReduceOp>(consumer))
    for (Operation &operation : body.without_terminator())
      if (operation.getNumRegions() || !isMemoryEffectFree(&operation))
        return std::nullopt;
  auto payload = analyzeProducerResult(producer, storage);
  if (failed(payload)) return std::nullopt;
  auto generic = dyn_cast<linalg::GenericOp>(consumer);
  ValueRange outputs = generic ? ValueRange(generic.getOutputs()) : ValueRange{};
  if (!canReplayProducerAt(*payload, producer, consumer, storage, outputs))
    return std::nullopt;
  return PointwiseProducer{producer, std::move(*payload), *projection};
}

struct ProducerFusion {
  Operation *consumer;
  AffineMap coordinates;
  SmallVector<unsigned> fusedInputs;
  SmallVector<linalg::IndexOp> indices;
  SmallVector<Value> inputs;
  SmallVector<AffineMap> maps;
  SmallVector<int64_t> iterationExtents;
  bool removesProducer;
};

FailureOr<ProducerFusion> planFusion(Operation *consumer, unsigned inputNumber,
                                    const PointwiseProducer &source) {
  auto producer = source.operation;
  auto generic = dyn_cast<linalg::GenericOp>(consumer);
  auto reduce = dyn_cast<ReduceOp>(consumer);
  ValueRange oldInputs = generic ? ValueRange(generic.getInputs()) : ValueRange(reduce.getInputs());
  SmallVector<AffineMap> oldMaps = generic ? generic.getIndexingMapsArray()
      : llvm::to_vector(llvm::map_range(reduce.getIndexingMaps(), [](Attribute a) {
          return cast<AffineMapAttr>(a).getValue();
        }));
  auto producerMaps = producer.getIndexingMapsArray();
  AffineMap coordinates = source.outputProjection.compose(oldMaps[inputNumber]);
  Value buffer = producer.getOutputs()[0];
  SmallVector<unsigned> fusedInputs;
  for (auto [number, input] : llvm::enumerate(oldInputs))
    if (input == buffer && oldMaps[number] == oldMaps[inputNumber])
      fusedInputs.push_back(number);
  bool removesProducer = llvm::all_of(buffer.getUses(), [&](OpOperand &use) {
    Operation *user = use.getOwner();
    return user == producer || isRemovableProducerMetadata(user) ||
        (user == consumer && llvm::is_contained(fusedInputs, use.getOperandNumber()));
  });
  // Reduce operands include extent and initial before their input segment.
  if (reduce)
    removesProducer = llvm::all_of(buffer.getUsers(), [&](Operation *user) {
      return user == producer || user == consumer || isRemovableProducerMetadata(user);
    }) && llvm::all_of(llvm::enumerate(oldInputs), [&](auto item) {
      return item.value() != buffer || llvm::is_contained(fusedInputs, item.index());
    });
  SmallVector<int64_t> iterationExtents;
  if (generic) iterationExtents = generic.getStaticLoopRanges();
  else {
    iterationExtents.assign(coordinates.getNumDims(), ShapedType::kDynamic);
    if (iterationExtents.size() == 1)
      if (auto extent = getConstantIntValue(reduce.getExtent())) iterationExtents[0] = *extent;
  }
  SmallVector<linalg::IndexOp> indices;
  producer.getRegion().walk([&](linalg::IndexOp index) {
    if (llvm::is_contained(source.payload.frontier, index.getResult())) indices.push_back(index);
  });
  if (reduce) {
    if (!indices.empty()) return failure();
    for (Operation &operation : producer.getRegion().front().without_terminator())
      if (operation.getNumRegions() || !isMemoryEffectFree(&operation)) return failure();
  }
  for (auto index : indices)
    if (!isa<AffineDimExpr, AffineConstantExpr>(coordinates.getResult(index.getDim())))
      return failure();
  SmallVector<Value> inputs;
  SmallVector<AffineMap> maps;
  for (auto [i, input] : llvm::enumerate(oldInputs)) {
    if (llvm::is_contained(fusedInputs, i)) {
      if (i != fusedInputs.front()) continue;
      for (auto [j, operand] : llvm::enumerate(producer.getInputs())) {
        if (!llvm::is_contained(source.payload.frontier,
                               producer.getRegion().front().getArgument(j))) continue;
        inputs.push_back(operand);
        maps.push_back(producerMaps[j].compose(coordinates));
      }
    } else {
      inputs.push_back(input);
      maps.push_back(oldMaps[i]);
    }
  }
  if (generic) {
    // Removing an input must not remove the only shape binding for a loop.
    llvm::SmallBitVector bound(generic.getNumLoops());
    for (AffineMap map : maps)
      for (AffineExpr expression : map.getResults())
        if (auto dimension = dyn_cast<AffineDimExpr>(expression)) bound.set(dimension.getPosition());
    for (AffineExpr expression : oldMaps.back().getResults())
      if (auto dimension = dyn_cast<AffineDimExpr>(expression)) bound.set(dimension.getPosition());
    if (!bound.all()) return failure();
    maps.push_back(oldMaps.back());
  }
  return ProducerFusion{consumer, coordinates, std::move(fusedInputs),
      std::move(indices), std::move(inputs), std::move(maps),
      std::move(iterationExtents), removesProducer};
}

void applyFusion(const ProducerFusion &plan, const PointwiseProducer &source) {
  Operation *consumer = plan.consumer;
  auto producer = source.operation;
  auto generic = dyn_cast<linalg::GenericOp>(consumer);
  auto reduce = dyn_cast<ReduceOp>(consumer);
  ValueRange oldInputs = generic ? ValueRange(generic.getInputs()) : ValueRange(reduce.getInputs());
  Value yielded = producer.getRegion().front().getTerminator()->getOperand(0);
  const auto &fusedInputs = plan.fusedInputs;
  const auto &inputs = plan.inputs;
  const auto &maps = plan.maps;
  auto populate = [&](OpBuilder &b, Block &target, Block &oldBody, unsigned prefix) {
    IRMapping mapping;
    unsigned current = prefix;
    Value produced;
    if (prefix) mapping.map(oldBody.getArgument(0), target.getArgument(0));
    for (unsigned i = 0; i < oldInputs.size(); ++i) {
      if (!llvm::is_contained(fusedInputs, i)) {
        mapping.map(oldBody.getArgument(prefix + i), target.getArgument(current++));
        continue;
      }
      if (produced) {
        mapping.map(oldBody.getArgument(prefix + i), produced);
        continue;
      }
      IRMapping producerMapping;
      Block &body = producer.getRegion().front();
      for (unsigned j = 0; j < producer.getInputs().size(); ++j)
        if (llvm::is_contained(source.payload.frontier, body.getArgument(j)))
          producerMapping.map(body.getArgument(j), target.getArgument(current++));
      for (auto index : plan.indices) {
        AffineExpr coordinate = plan.coordinates.getResult(index.getDim());
        Value value;
        if (auto dimension = dyn_cast<AffineDimExpr>(coordinate))
          value = b.create<linalg::IndexOp>(index.getLoc(), dimension.getPosition());
        else value = b.create<arith::ConstantIndexOp>(index.getLoc(), cast<AffineConstantExpr>(coordinate).getValue());
        producerMapping.map(index.getResult(), value);
      }
      auto cloned = materializeProducerValue(source.payload, yielded, b, producerMapping);
      assert(succeeded(cloned) && "structured replay must bind the complete payload frontier");
      produced = *cloned;
      mapping.map(oldBody.getArgument(prefix + i), produced);
    }
    if (generic)
      mapping.map(oldBody.getArguments().back(), target.getArguments().back());
    for (Operation &operation : oldBody.without_terminator()) b.clone(operation, mapping);
    return mapping.lookupOrDefault(oldBody.getTerminator()->getOperand(0));
  };
  OpBuilder builder(consumer);
  if (generic) {
    auto replacement = builder.create<linalg::GenericOp>(generic.getLoc(), inputs,
        generic.getOutputs(), maps, generic.getIteratorTypesArray(),
        [](OpBuilder &, Location, ValueRange) {});
    Block &body = replacement.getRegion().front();
    builder.setInsertionPointToStart(&body);
    Value value = populate(builder, body, generic.getRegion().front(), 0);
    builder.create<linalg::YieldOp>(generic.getLoc(), value);
    replacement->setDiscardableAttrs(llvm::to_vector(generic->getDiscardableAttrs()));
    generic.erase();
  } else {
    SmallVector<Attribute> attributes;
    for (AffineMap map : maps) attributes.push_back(AffineMapAttr::get(map));
    auto replacement = builder.create<ReduceOp>(reduce.getLoc(), reduce.getResult().getType(),
        reduce.getExtent(), reduce.getInitial(), inputs, builder.getArrayAttr(attributes),
        reduce.getOrder());
    replacement->setDiscardableAttrs(llvm::to_vector(reduce->getDiscardableAttrs()));
    Block &body = replacement.getCombine().emplaceBlock();
    body.addArgument(reduce.getInitial().getType(), reduce.getLoc());
    for (Value input : inputs) {
      auto memory = dyn_cast<MemRefType>(input.getType());
      body.addArgument(memory ? memory.getElementType() : input.getType(), reduce.getLoc());
    }
    builder.setInsertionPointToStart(&body);
    Value value = populate(builder, body, reduce.getCombine().front(), 1);
    builder.create<YieldOp>(reduce.getLoc(), value);
    reduce.getResult().replaceAllUsesWith(replacement.getResult());
    reduce.erase();
  }
}

bool fuseOne(Operation *consumer, unsigned inputNumber,
             const PointwiseProducer &source) {
  auto plan = planFusion(consumer, inputNumber, source);
  if (failed(plan)) return false;
  auto producer = source.operation;
  Value yielded = producer.getRegion().front().getTerminator()->getOperand(0);
  if (!shouldFuseProducerResult(source.payload, yielded, plan->coordinates,
                                plan->iterationExtents, plan->removesProducer))
    return false;
  applyFusion(*plan, source);
  eraseUnusedProducer(source.operation);
  return true;
}

bool fuseOneVersion(func::FuncOp function) {
  SmallVector<linalg::GenericOp> producers;
  function.walk([&](linalg::GenericOp producer) {
    if (!producer.getNumResults() && producer.getOutputs().size() == 1 &&
        !producer.getNumReductionLoops()) producers.push_back(producer);
  });
  StorageAnalysis storage(function);
  for (auto producer : producers) {
    auto version = queryProducerVersion(producer, storage);
    if (failed(version)) continue;
    auto payload = analyzeProducerResult(producer, storage);
    auto projection = fullOutputProjection(producer.getOutputs()[0],
                                           producer.getIndexingMapsArray().back());
    if (failed(payload) || failed(projection)) continue;
    PointwiseProducer source{producer, std::move(*payload), *projection};
    SmallVector<ProducerFusion, 2> plans;
    bool legal = true;
    Value yielded = producer.getRegion().front().getTerminator()->getOperand(0);
    for (const ProducerVersionUse &use : version->uses) {
      if (!fusionConsumer(use.operation, storage)) { legal = false; break; }
      auto plan = planFusion(use.operation, use.input, source);
      if (failed(plan) ||
          !canReplayProducerVersionAt(source.payload, producer, use.operation,
                                      plan->coordinates, storage) ||
          !shouldFuseProducerResult(source.payload, yielded, plan->coordinates,
                                    plan->iterationExtents, true, version->uses.size())) {
        legal = false;
        break;
      }
      plans.push_back(std::move(*plan));
    }
    if (!legal) continue;
    // Earlier consumers can define scalar results captured by later ones.
    // Rebuild backwards so those original values remain live; their later RAUW
    // then updates the already-published consumer without stale plan operands.
    for (const ProducerFusion &plan : llvm::reverse(plans)) applyFusion(plan, source);
    producer.erase();
    return true; // No query uses this storage snapshot after the first rewrite.
  }
  return false;
}

bool reuseOutput(linalg::GenericOp consumer, Value buffer,
                 StorageAnalysis &storage) {
  auto allocation = buffer.getDefiningOp<memref::AllocOp>();
  if (!allocation || allocation->getBlock() != consumer->getBlock() ||
      consumer.getNumReductionLoops() || consumer.getOutputs().size() != 1 ||
      !consumer.getRegion().front().getArguments().back().use_empty()) return false;
  auto maps = consumer.getIndexingMapsArray();
  if (!maps.back().isIdentity()) return false;
  for (auto [input, map] : llvm::zip(consumer.getInputs(), maps))
    if (input == buffer && !map.isIdentity()) return false;
  Value output = consumer.getOutputs()[0];
  auto external = storage.externalView(output);
  auto view = output.getDefiningOp<memref::SubViewOp>();
  if (!external || external.getAccess() != 1 || !view ||
      view.getType().getRank() != allocation.getType().getRank() ||
      view.getType().getElementType() != allocation.getType().getElementType()) return false;
  SmallVector<OpFoldResult> extents;
  auto dropped = view.getDroppedDims();
  for (auto [axis, size] : llvm::enumerate(view.getMixedSizes()))
    if (!dropped.test(axis)) extents.push_back(size);
  unsigned dynamic = 0;
  for (auto [axis, extent] : llvm::enumerate(extents)) {
    if (allocation.getType().isDynamicDim(axis)) {
      Value size = allocation.getDynamicSizes()[dynamic++];
      if (extent == OpFoldResult(size)) continue;
      auto left = getConstantIntValue(extent), right = getConstantIntValue(size);
      if (!left || !right || *left != *right) return false;
    } else if (getConstantIntValue(extent) != allocation.getType().getDimSize(axis)) return false;
  }
  auto aliases = storage.aliases(output);
  if (!aliases.complete || llvm::any_of(aliases.users, [&](Operation *user) {
        return user != consumer && !isa<memref::DimOp>(user) && !isStorageAliasOperation(user);
      })) return false;
  linalg::GenericOp producer;
  llvm::SmallPtrSet<Operation *, 4> readers;
  SmallVector<memref::DeallocOp> deallocations;
  for (Operation *user : buffer.getUsers()) {
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      deallocations.push_back(dealloc);
      continue;
    }
    if (isa<memref::DimOp>(user)) continue;
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (generic && llvm::is_contained(generic.getOutputs(), buffer)) {
      if (producer || generic.getOutputs().size() != 1 || generic.getNumReductionLoops() ||
          !generic.getIndexingMapsArray().back().isIdentity() ||
          !generic.getRegion().front().getArguments().back().use_empty() ||
          llvm::is_contained(generic.getInputs(), buffer)) return false;
      producer = generic;
    } else if (generic || isa<ReduceOp>(user)) readers.insert(user);
    else return false;
    if (user->getBlock() != consumer->getBlock() ||
        (user != consumer && !user->isBeforeInBlock(consumer))) return false;
  }
  if (!producer || readers.size() < 2) return false;
  for (Operation *reader : readers)
    if (!producer->isBeforeInBlock(reader)) return false;
  for (auto operation : {producer, consumer})
    for (Operation &nested : operation.getRegion().front().without_terminator())
      if (nested.getNumRegions() || !isMemoryEffectFree(&nested)) return false;
  DominanceInfo dominance(consumer->getParentOfType<func::FuncOp>());
  if (!dominance.dominates(output, allocation)) {
    if (view->getBlock() != allocation->getBlock() ||
        llvm::any_of(view->getOperands(), [&](Value value) {
          return !dominance.dominates(value, allocation);
        })) return false;
    view->moveBefore(allocation);
  }
  for (memref::DeallocOp dealloc : deallocations) dealloc.erase();
  allocation.getResult().replaceAllUsesWith(output);
  allocation.erase();
  return true;
}

void forwardPointwiseCopies(func::FuncOp function) {
  SmallVector<memref::AllocOp> allocations;
  function.walk([&](memref::AllocOp allocation) { allocations.push_back(allocation); });
  for (auto allocation : allocations) {
    Value buffer = allocation.getResult();
    llvm::DenseMap<Block *, linalg::GenericOp> writers;
    llvm::DenseMap<Block *, memref::CopyOp> readers;
    bool closed = true;
    for (Operation *user : buffer.getUsers()) {
      if (isa<memref::DimOp, memref::DeallocOp>(user)) continue;
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getSource() != buffer || copy.getTarget() == buffer ||
            !readers.try_emplace(copy->getBlock(), copy).second) closed = false;
        continue;
      }
      auto writer = dyn_cast<linalg::GenericOp>(user);
      if (!writer || writer.getNumResults() || writer.getNumReductionLoops() ||
          writer.getOutputs().size() != 1 || writer.getOutputs()[0] != buffer ||
          llvm::is_contained(writer.getInputs(), buffer) ||
          !writer.getIndexingMapsArray().back().isIdentity() ||
          !writer.getRegion().front().getArguments().back().use_empty() ||
          llvm::any_of(writer.getRegion().front().without_terminator(), [](Operation &operation) {
            return operation.getNumRegions() || !isMemoryEffectFree(&operation);
          }) || !writers.try_emplace(writer->getBlock(), writer).second) closed = false;
    }
    if (!closed || readers.empty() || readers.size() != writers.size()) continue;
    // Every read has its own complete same-block definition, including distinct
    // full/tail loop bodies. No other user may observe this scratch afterwards.
    if (llvm::any_of(readers, [&](auto &entry) {
          auto writer = writers.lookup(entry.first);
          return !writer || !writer->isBeforeInBlock(entry.second);
        })) continue;
    for (auto &entry : readers) {
      auto copy = entry.second;
      auto writer = writers.lookup(entry.first);
      Value target = copy.getTarget();
      StorageAnalysis storage(function);
      Value targetRoot = storage.uniqueOrigin(target);
      // Retain a private owner. Forwarding into caller storage also needs the
      // provider's native ABI to preserve the disjointness used for load reuse.
      if (!targetRoot ||
          !isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(targetRoot.getDefiningOp())) continue;
      DominanceInfo dominance(function);
      if (!dominance.dominates(target, writer)) continue;
      bool legal = true;
      auto strip = [](Value value) {
        while (auto cast = value.getDefiningOp<memref::CastOp>()) value = cast.getSource();
        return value;
      };
      auto maps = writer.getIndexingMapsArray();
      for (auto [number, input] : llvm::enumerate(writer.getInputs())) {
        if (!isa<MemRefType>(input.getType()) || storage.disjoint(input, target)) continue;
        if (strip(input) != strip(target) || maps[number] != maps.back()) legal = false;
      }
      // Only dst[i] = f(dst[i], ...) is safe in-place. Other observations of the
      // old destination between the definition and its copy keep the snapshot.
      for (Operation *between = writer->getNextNode(); legal && between != copy;
           between = between->getNextNode()) {
        auto effects = storage.effects(between);
        if (!effects.complete || effects.ordered) { legal = false; break; }
        for (const StorageEffect &entry : effects.entries) {
          const auto &effect = entry.effect;
          if (isa<MemoryEffects::Allocate>(effect.getEffect())) continue;
          if (!effect.getValue() || !storage.disjoint(effect.getValue(), target)) legal = false;
        }
      }
      if (!legal) continue;
      writer.getDpsInitsMutable()[0].set(target);
      copy.erase();
    }
  }
}

}

LogicalResult fuseStructuredComputations(func::FuncOp function) {
  if (failed(foldIntegerSources(function))) return failure();
  foldContiguousAccesses(function);
  forwardCPUOutputs(function);
  bool changed;
  do {
    changed = forwardProducerVersionCopies(function);
    if (changed) continue;
    changed = fuseOneVersion(function);
    if (changed) continue;
    SmallVector<Operation *> consumers;
    StorageAnalysis payloadStorage(function);
    function.walk([&](Operation *operation) {
      if (fusionConsumer(operation, payloadStorage)) consumers.push_back(operation);
    });
    for (Operation *consumer : llvm::reverse(consumers)) {
      StorageAnalysis analysis(function);
      auto generic = dyn_cast<linalg::GenericOp>(consumer);
      ValueRange inputs = generic ? ValueRange(generic.getInputs())
                                 : ValueRange(cast<ReduceOp>(consumer).getInputs());
      for (auto [number, input] : llvm::enumerate(inputs)) {
        auto producer = pointwiseProducer(input, consumer, analysis);
        if (!producer) continue;
        changed = fuseOne(consumer, number, *producer);
        if (changed) break;
      }
      if (changed) break;
    }
  } while (changed);
  SmallVector<linalg::GenericOp> outputs;
  function.walk([&](linalg::GenericOp operation) { outputs.push_back(operation); });
  for (auto operation : llvm::reverse(outputs)) {
    SmallVector<Value> inputs(operation.getInputs());
    for (Value input : inputs) {
      StorageAnalysis analysis(function);
      if (reuseOutput(operation, input, analysis)) break;
    }
  }
  forwardPointwiseCopies(function);
  eraseDeadPrivateBuffers(function);
  return success();
}


}
