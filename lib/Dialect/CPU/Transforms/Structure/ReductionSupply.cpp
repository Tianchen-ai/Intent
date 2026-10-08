#include "ReductionSupply.h"
#include "ReductionSources.h"
#include "ProducerReuse.h"
#include "../Vector/ContiguousMemory.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

using namespace mlir;

namespace intent::cpu {
namespace {

struct SupplyGroup {
  Value buffer;
  Value input;
  SmallVector<Operation *> readers;
  Operation *last;
};

bool identityInput(Value memory, AffineMap map) {
  auto type = dyn_cast<MemRefType>(memory.getType());
  return type ? map.isIdentity() : map.getNumResults() == 0;
}

// A complete collapse retains the member order and its original dimension
// values. Restore those dimensions instead of inventing lane-wise div/mod.
Value fullCollapsedSource(Value value) {
  auto collapse = value.getDefiningOp<memref::CollapseShapeOp>();
  if (!collapse || collapse.getType().getRank() != 1 ||
      collapse.getReassociationIndices().size() != 1) return {};
  return collapse.getSrc();
}

bool canRestore(ReduceOp reduce, Value buffer) {
  auto order = reduce.getOrder();
  if (!order.getAdjacentReassociation() || !order.getElementPermutation()) return false;
  auto maps = reduce.getIndexingMaps();
  bool found = false;
  for (auto [number, input] : llvm::enumerate(reduce.getInputs())) {
    auto map = cast<AffineMapAttr>(maps[number]).getValue();
    if (!isa<MemRefType>(input.getType())) {
      if (map.getNumResults()) return false;
      continue;
    }
    Value source = fullCollapsedSource(input);
    if (!source || !map.isIdentity()) return false;
    auto type = cast<MemRefType>(source.getType());
    auto rootType = cast<MemRefType>(buffer.getType());
    if (type.getRank() != rootType.getRank() ||
        !haveEqualExtents(ValueBoundsConstraintSet::Variable(reduce.getExtent()),
                          ValueBoundsConstraintSet::Variable(input, 0))) return false;
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      if (!haveEqualExtents(ValueBoundsConstraintSet::Variable(source, axis),
                            ValueBoundsConstraintSet::Variable(buffer, axis))) return false;
    found |= source == buffer;
  }
  return found;
}

std::optional<SupplyGroup> queryGroup(linalg::GenericOp producer) {
  auto loop = dyn_cast<scf::ForOp>(producer->getParentOp());
  if (!loop || producer.getNumResults() || producer.getNumReductionLoops() ||
      producer.getInputs().size() != 1 || producer.getOutputs().size() != 1 ||
      !producer.getRegion().front().getArguments().back().use_empty()) return std::nullopt;
  Value buffer = producer.getOutputs()[0], input = producer.getInputs()[0];
  auto allocation = buffer.getDefiningOp<memref::AllocOp>();
  auto inputType = dyn_cast<MemRefType>(input.getType());
  auto outputType = dyn_cast<MemRefType>(buffer.getType());
  auto binding = producer->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  if (!allocation || !inputType || !outputType || !inputType.getRank() ||
      inputType.getRank() != outputType.getRank() || !binding ||
      !binding.getParameters().getAs<IntegerAttr>("vector_width")) return std::nullopt;
  if (binding.getParameters().getAs<IntegerAttr>("vector_width").getInt() <= 1)
    return std::nullopt;
  for (AffineMap map : producer.getIndexingMapsArray())
    if (!map.isIdentity()) return std::nullopt;
  for (int64_t axis = 0; axis < inputType.getRank(); ++axis)
    if (!haveEqualExtents(ValueBoundsConstraintSet::Variable(input, axis),
                          ValueBoundsConstraintSet::Variable(buffer, axis))) return std::nullopt;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(inputType.getStridesAndOffset(strides, offset)) ||
      (strides.back() != 1 && !ShapedType::isDynamic(strides.back()))) return std::nullopt;
  StorageAnalysis storage(producer->getParentOfType<func::FuncOp>());
  if (!completelyWritesBuffer(producer, buffer)) return std::nullopt;
  auto payload = analyzeProducerResult(producer, storage);
  if (failed(payload) || convertedLoadSource(*payload,
          producer.getRegion().front().getTerminator()->getOperand(0)) !=
          producer.getRegion().front().getArgument(0)) return std::nullopt;
  auto version = queryBufferVersionObservations(buffer, producer, storage);
  if (failed(version) || version->reads.empty() ||
      version->end != loop.getBody()->getTerminator()) return std::nullopt;
  SupplyGroup group{buffer, input, version->reads, producer};
  for (Operation *reader : group.readers) {
    if (reader->getBlock() != producer->getBlock() || !producer->isBeforeInBlock(reader) ||
        reader->getAttr("intent_cpu.implementation") != binding) return std::nullopt;
    ValueRange outputs;
    if (auto generic = dyn_cast<linalg::GenericOp>(reader)) {
      if (generic.getNumResults() || !generic.getNumReductionLoops()) return std::nullopt;
      bool found = false;
      auto maps = generic.getIndexingMapsArray();
      for (auto [number, operand] : llvm::enumerate(generic.getInputs())) {
        if (!isa<MemRefType>(operand.getType())) continue;
        if (storage.disjoint(operand, buffer)) continue;
        if (operand != buffer || !identityInput(operand, maps[number])) return std::nullopt;
        found = true;
      }
      if (!found) return std::nullopt;
      outputs = generic.getOutputs();
    } else if (auto reduce = dyn_cast<ReduceOp>(reader)) {
      if (!canRestore(reduce, buffer)) return std::nullopt;
    } else return std::nullopt;
    if (!canReplayProducerAt(*payload, producer, reader, storage, outputs)) return std::nullopt;
    if (group.last->isBeforeInBlock(reader)) group.last = reader;
  }
  return group;
}

linalg::GenericOp restore(ReduceOp reduce) {
  OpBuilder builder(reduce);
  Location loc = reduce.getLoc();
  SmallVector<Value> inputs;
  unsigned rank = 0;
  for (Value input : reduce.getInputs()) {
    if (isa<MemRefType>(input.getType())) {
      input = fullCollapsedSource(input);
      rank = cast<MemRefType>(input.getType()).getRank();
    }
    inputs.push_back(input);
  }
  Value output = builder.create<memref::AllocaOp>(loc,
      MemRefType::get({}, reduce.getResult().getType()));
  builder.create<memref::StoreOp>(loc, reduce.getInitial(), output, ValueRange{});
  SmallVector<AffineMap> maps;
  for (Value input : inputs)
    maps.push_back(isa<MemRefType>(input.getType()) ? builder.getMultiDimIdentityMap(rank)
        : AffineMap::get(rank, 0, {}, builder.getContext()));
  maps.push_back(AffineMap::get(rank, 0, {}, builder.getContext()));
  auto generic = builder.create<linalg::GenericOp>(loc, inputs, ValueRange{output}, maps,
      SmallVector<utils::IteratorType>(rank, utils::IteratorType::reduction),
      [&](OpBuilder &nested, Location at, ValueRange arguments) {
        IRMapping mapping;
        Block &body = reduce.getCombine().front();
        mapping.map(body.getArgument(0), arguments.back());
        mapping.map(body.getArguments().drop_front(), arguments.drop_back());
        for (Operation &operation : body.without_terminator()) nested.clone(operation, mapping);
        nested.create<linalg::YieldOp>(at,
            mapping.lookupOrDefault(body.getTerminator()->getOperand(0)));
      });
  generic->setDiscardableAttrs(llvm::to_vector(reduce->getDiscardableAttrs()));
  generic->setAttr("intent_cpu.reduction_order", reduce.getOrder());
  Value result = builder.create<memref::LoadOp>(loc, output, ValueRange{});
  reduce.getResult().replaceAllUsesWith(result);
  reduce.erase();
  return generic;
}

} // namespace

bool hasReductionSupplyGroup(linalg::GenericOp producer) {
  return queryGroup(producer).has_value();
}

FailureOr<bool> materializeReductionSupplyGroup(linalg::GenericOp producer, int64_t width,
    llvm::function_ref<LogicalResult(linalg::GenericOp, Value, ArrayRef<Operation *>)> materialize) {
  if (width <= 1) return false;
  auto group = queryGroup(producer);
  if (!group) return false;
  SmallVector<Operation *> interval;
  for (Operation *operation = producer;; operation = operation->getNextNode()) {
    interval.push_back(operation);
    if (operation == group->last) break;
  }
  auto inside = [&](Operation *user) {
    return llvm::any_of(interval, [&](Operation *operation) {
      return operation == user || operation->isAncestor(user);
    });
  };
  SmallVector<Value> live;
  for (Operation *operation : interval)
    for (Value result : operation->getResults())
      if (llvm::any_of(result.getUsers(), [&](Operation *user) { return !inside(user); }))
        live.push_back(result);
  OpBuilder builder(producer);
  Location loc = producer.getLoc();
  Operation *previous = producer->getPrevNode();
  auto guard = materializeContiguousMemoryGuard(builder, loc, ValueRange{group->input});
  Value condition = guard.condition;
  if (!condition) condition = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  auto choice = builder.create<scf::IfOp>(loc, TypeRange(live), condition, true);
  auto finish = [&](Block *block, ValueRange results) {
    if (!block->empty())
      if (auto yield = dyn_cast<scf::YieldOp>(block->back())) {
        yield->setOperands(results);
        return;
      }
    builder.create<scf::YieldOp>(loc, results);
  };
  IRMapping fast;
  builder.setInsertionPointToStart(choice.thenBlock());
  guard.bind(builder, loc, fast);
  auto allocation = group->buffer.getDefiningOp<memref::AllocOp>();
  Value local = builder.clone(*allocation)->getResult(0);
  fast.map(group->buffer, local);
  for (Operation *reader : group->readers)
    if (auto reduce = dyn_cast<ReduceOp>(reader))
      for (Value input : reduce.getInputs())
        if (fullCollapsedSource(input) == group->buffer && !fast.contains(input))
          builder.clone(*input.getDefiningOp(), fast);
  for (Operation *operation : interval) builder.clone(*operation, fast);
  builder.create<memref::DeallocOp>(loc, local);
  SmallVector<Value> fastResults;
  for (Value result : live) fastResults.push_back(fast.lookupOrDefault(result));
  finish(choice.thenBlock(), fastResults);
  builder.setInsertionPointToStart(choice.elseBlock());
  IRMapping original;
  for (Operation *operation : interval) builder.clone(*operation, original);
  SmallVector<Value> originalResults;
  for (Value result : live) originalResults.push_back(original.lookupOrDefault(result));
  finish(choice.elseBlock(), originalResults);

  SmallVector<Operation *> readers;
  for (Operation *reader : group->readers) {
    Operation *cloned = fast.lookup(reader);
    if (auto reduce = dyn_cast<ReduceOp>(cloned)) cloned = restore(reduce);
    readers.push_back(cloned);
  }
  for (Operation *operation : readers) {
    auto reader = cast<linalg::GenericOp>(operation);
    ReductionSources members(reader, local, readers);
    for (auto [number, input] : llvm::enumerate(reader.getInputs()))
      if (input == local && !members.replays(number)) {
        choice.erase();
        while (producer->getPrevNode() != previous)
          producer->getPrevNode()->erase();
        return false;
      }
  }
  // All readers and both branches are complete before the fresh storage queries
  // used by member replay. There is no analysis spanning an attached rewrite.
  while (!readers.empty()) {
    auto reader = cast<linalg::GenericOp>(readers.front());
    if (failed(materialize(reader, local, readers))) return failure();
    readers.erase(readers.begin());
  }
  for (auto [result, replacement] : llvm::zip(live, choice.getResults()))
    result.replaceAllUsesWith(replacement);
  for (Operation *operation : llvm::reverse(interval)) operation->erase();
  return true;
}

} // namespace intent::cpu
