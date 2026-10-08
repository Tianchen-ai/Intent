#include "Intent/Dialect/CPU/Transforms/Vector/Vectorization.h"
#include "ContiguousMemory.h"
#include "ProducerVectorization.h"
#include "../Storage/AccessAliases.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "Intent/Dialect/CPU/Transforms/Vector/VectorReductions.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool contiguousStore(Value memory, ValueRange indices, scf::ForOp loop,
                     const ProducerVectorization &producer,
                     SmallVectorImpl<Value> &guardedMemories) {
  auto type = cast<MemRefType>(memory.getType());
  if (!ProducerVectorization::isElementType(type.getElementType()) ||
      type.getElementType().isInteger(1)) return false;
  if (auto owner = memory.getDefiningOp(); owner && loop->isAncestor(owner)) return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset))) return false;
  bool invariant = true;
  for (auto [axis, index] : llvm::enumerate(indices)) {
    auto c = producer.coefficient(index);
    if (!c) return false;
    if (*c != 0) {
      invariant = false;
      if (*c != 1 || axis + 1 != indices.size()) return false;
      if (ShapedType::isDynamic(strides[axis])) {
        if (!llvm::is_contained(guardedMemories, memory)) guardedMemories.push_back(memory);
      } else if (strides[axis] != 1) return false;
    }
  }
  return !invariant;
}

void vectorize(scf::ForOp original, int64_t width, int64_t replicas, bool nonempty = false) {
  if (!matchPattern(original.getStep(), m_One())) return;
  SmallVector<Value> reductionInputs;
  SmallVector<Operation *> combines;
  SmallVector<IndependentReduction> nativeReductions;
  if (original.getNumResults()) {
    auto order = original->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
    if (!order || !order.getAdjacentReassociation()) return;
    for (auto [carry, yielded] : llvm::zip(original.getRegionIterArgs(),
             original.getBody()->getTerminator()->getOperands())) {
      if (!carry.getType().isF32() || !carry.hasOneUse()) return;
      Operation *combine = yielded.getDefiningOp();
      if (!combine || !isa<arith::AddFOp, arith::MulFOp, arith::MaxNumFOp, arith::MaximumFOp, arith::MinimumFOp>(combine)) return;
      if (combine->getOperand(0) == carry) reductionInputs.push_back(combine->getOperand(1));
      else if (combine->getOperand(1) == carry) reductionInputs.push_back(combine->getOperand(0));
      else return;
      combines.push_back(combine);
    }
    if (order.getElementPermutation()) {
      for (auto [number, carry] : llvm::enumerate(original.getRegionIterArgs())) {
        auto reduction = matchIndependentReduction(carry, reductionInputs[number],
            original.getBody()->getTerminator()->getOperand(number));
        if (!reduction) { nativeReductions.clear(); break; }
        nativeReductions.push_back(*reduction);
      }
    }
  }
  SmallVector<memref::StoreOp> stores;
  SmallVector<memref::LoadOp> loads;
  SmallVector<Value> guardedMemories;
  llvm::SmallSetVector<Value, 32> roots;
  for (Operation &operation : original.getBody()->without_terminator()) {
    if (auto store = dyn_cast<memref::StoreOp>(&operation)) {
      stores.push_back(store);
      roots.insert(store.getValue());
      roots.insert(store.getIndices().begin(), store.getIndices().end());
    } else if (!llvm::is_contained(combines, &operation)) {
      if (!operation.getNumResults() ||
          !llvm::all_of(operation.getResultTypes(), ProducerVectorization::isElementType))
        return;
      roots.insert(operation.getResults().begin(), operation.getResults().end());
    }
  }
  if (reductionInputs.empty() && stores.empty()) return;
  auto function = original->getParentOfType<func::FuncOp>();
  StorageAnalysis storage(function);
  roots.insert(reductionInputs.begin(), reductionInputs.end());
  SmallVector<Value> frontier{original.getInductionVar()};
  llvm::append_range(frontier, original.getRegionIterArgs());
  ProducerReplay payload{original, frontier, {}, {}, {}, {}};
  auto appendUnique = [](auto &target, const auto &values) {
    for (auto value : values)
      if (!llvm::is_contained(target, value)) target.push_back(value);
  };
  for (Value value : roots) {
    auto proof = analyzeProducerValue(value, original, frontier, storage);
    if (failed(proof)) return;
    appendUnique(payload.externalValues, proof->externalValues);
    appendUnique(payload.reads, proof->reads);
    appendUnique(payload.operations, proof->operations);
    appendUnique(payload.nodes, proof->nodes);
  }
  ProducerVectorization producer(payload, original.getInductionVar());
  for (Value value : roots) {
    if (llvm::any_of(original.getRegionIterArgs(), [&](Value carry) {
          return producer.dependsOn(value, carry);
        }) || !producer.canWiden(value, &guardedMemories)) return;
  }
  // Carries participate only in the separately preserved reduction combiner.
  // Every widened producer has been proved independent of those formals.
  payload.frontier.resize(1);
  for (Operation *operation : payload.nodes)
    if (auto load = dyn_cast<memref::LoadOp>(operation)) loads.push_back(load);
  for (auto store : stores)
    if (!contiguousStore(store.getMemref(), store.getIndices(), original,
                         producer, guardedMemories)) return;
  auto independent = [&](Value lhs, ValueRange lhsIndices, Value rhs, ValueRange rhsIndices) {
    if (lhs == rhs && lhsIndices == rhsIndices) return true;
    return storage.disjointAt(lhs, rhs, original);
  };
  for (auto [number, store] : llvm::enumerate(stores)) {
    for (auto load : loads)
      if (!independent(load.getMemref(), load.getIndices(), store.getMemref(), store.getIndices())) return;
    for (auto other : ArrayRef(stores).drop_front(number + 1))
      if (!independent(other.getMemref(), other.getIndices(), store.getMemref(), store.getIndices())) return;
  }
  OpBuilder b(original);
  Location loc = original.getLoc();
  bool needsInvariantGuard = llvm::any_of(original.getBody()->without_terminator(), [&](Operation &op) {
    return !isSpeculatable(&op) && llvm::any_of(op.getResults(), [&](Value value) {
      return producer.isUniform(value);
    });
  });
  if (!guardedMemories.empty() || (!nonempty && needsInvariantGuard)) {
    auto contiguous = materializeContiguousMemoryGuard(b, loc, guardedMemories);
    Value condition = contiguous.condition;
    if (!nonempty && needsInvariantGuard) {
      Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
          original.getUpperBound(), original.getLowerBound());
      condition = condition ? Value(b.create<arith::AndIOp>(loc, condition, active)) : active;
    }
    auto dispatch = b.create<scf::IfOp>(loc, original.getResultTypes(), condition, true);
    original.replaceAllUsesWith(dispatch.getResults());
    b.setInsertionPointToStart(&dispatch.getThenRegion().front());
    IRMapping mapping;
    contiguous.bind(b, loc, mapping);
    auto contiguousLoop = cast<scf::ForOp>(b.clone(*original, mapping));
    if (original.getNumResults()) {
      b.setInsertionPointToEnd(&dispatch.getThenRegion().front());
      b.create<scf::YieldOp>(loc, contiguousLoop.getResults());
    }
    original->moveBefore(&dispatch.getElseRegion().front(), dispatch.getElseRegion().front().begin());
    if (original.getNumResults()) {
      b.setInsertionPointToEnd(&dispatch.getElseRegion().front());
      b.create<scf::YieldOp>(loc, original.getResults());
    }
    // Recursive vectorization queries the complete function's storage flow.
    // Finish both successor regions before taking that new analysis snapshot.
    vectorize(contiguousLoop, width, replicas, nonempty || needsInvariantGuard);
    return;
  }
  int64_t logicalWidth = replicas * width;
  Value step = index(b, loc, logicalWidth);
  Value length = b.create<arith::SubIOp>(loc, original.getUpperBound(), original.getLowerBound());
  auto merge = [&](Operation *combine, Value lhs, Value rhs) {
    IRMapping mapping;
    mapping.map(combine->getOperand(0), lhs);
    mapping.map(combine->getOperand(1), rhs);
    Operation *result = b.clone(*combine, mapping);
    result->getResult(0).setType(lhs.getType());
    return result->getResult(0);
  };
  auto combineValues = [&](ValueRange lhs, ValueRange rhs, int64_t) {
    SmallVector<Value> results;
    for (auto [number, combine] : llvm::enumerate(combines))
      results.push_back(merge(combine, lhs[number], rhs[number]));
    return results;
  };
  auto vectorBody = [&](Value coordinate, OpBuilder &invariants) {
    IRMapping scalars, vectors;
    scalars.map(original.getInductionVar(), coordinate);
    auto vector = [&](Value value) {
      auto result = producer.materialize(value, logicalWidth, b, scalars,
                                         vectors, &invariants);
      assert(succeeded(result) && "loop widening must bind the proven producer");
      return *result;
    };
    auto indices = [&](ValueRange values) {
      SmallVector<Value> result;
      for (Value value : values) {
        auto index = producer.materializeScalar(value, b, scalars, &invariants);
        assert(succeeded(index) && "loop widening must bind the proven coordinate");
        result.push_back(*index);
      }
      return result;
    };
    for (Operation &operation : original.getBody()->without_terminator()) {
      if (auto load = dyn_cast<memref::LoadOp>(&operation)) vector(load.getResult());
      else if (auto branch = dyn_cast<scf::IfOp>(&operation))
        for (Value result : branch.getResults()) vector(result);
      else if (auto store = dyn_cast<memref::StoreOp>(&operation))
        b.create<vector::StoreOp>(loc, vector(store.getValue()), store.getMemref(),
                                  indices(store.getIndices()), store.getNontemporal());
    }
    SmallVector<Value> inputs;
    for (Value input : reductionInputs) inputs.push_back(vector(input));
    return inputs;
  };
  auto order = original->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  if (!combines.empty() && order.getElementPermutation()) {
    Value full = add(b, loc, original.getLowerBound(),
        multiply(b, loc, b.create<arith::DivSIOp>(loc, length, step), step));
    Value hasBlock = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                          full, original.getLowerBound());
    auto active = b.create<scf::IfOp>(loc, original.getResultTypes(), hasBlock, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(active.thenBlock());
      Value begin = add(b, loc, original.getLowerBound(), step);
      OpBuilder invariants(begin.getDefiningOp());
      // Seed each lane from an actual member. The original accumulator may be
      // a non-identity partial from an enclosing block and is included once.
      auto seeds = vectorBody(original.getLowerBound(), invariants);
      auto blocks = b.create<scf::ForOp>(loc, begin, full, step, seeds);
      {
        OpBuilder::InsertionGuard blockGuard(b);
        b.setInsertionPointToStart(blocks.getBody());
        auto values = vectorBody(blocks.getInductionVar(), invariants);
        b.create<scf::YieldOp>(loc, combineValues(blocks.getRegionIterArgs(), values, logicalWidth));
      }
      auto partial = horizontalReduce(b, loc, blocks.getResults(), combineValues, nativeReductions);
      b.create<scf::YieldOp>(loc, combineValues(original.getInitArgs(), partial, 0));
      b.setInsertionPointToStart(active.elseBlock());
      b.create<scf::YieldOp>(loc, original.getInitArgs());
    }
    original.setLowerBound(full);
    original.getInitArgsMutable().assign(active.getResults());
    if (replicas > 1) vectorize(original, width, 1);
    original->removeAttr("intent_cpu.reduction_order");
    return;
  }
  // Adjacent-only reductions keep every chunk and partial in coordinate order.
  // Register replicas widen a local tree without introducing striped carries.
  // Shorten the scalar carry chains with contiguous sums, then combine their
  // results in coordinate order. Splitting requires the closed +0 identity.
  int64_t partitions = stores.empty() && !combines.empty() &&
      llvm::all_of(combines, [](Operation *op) { return isa<arith::AddFOp>(op); }) &&
      llvm::all_of(original.getInitArgs(), [](Value value) { return matchPattern(value, m_PosZeroFloat()); })
      ? std::min<int64_t>(replicas, 4) : 1;
  Value groupStep = index(b, loc, logicalWidth * partitions);
  Value full = add(b, loc, original.getLowerBound(),
      multiply(b, loc, b.create<arith::DivSIOp>(loc, length, groupStep), groupStep));
  Value span = b.create<arith::DivSIOp>(loc,
      b.create<arith::SubIOp>(loc, full, original.getLowerBound()), index(b, loc, partitions));
  SmallVector<Value> initial;
  for (int64_t part = 0; part < partitions; ++part) llvm::append_range(initial, original.getInitArgs());
  auto vectorLoop = b.create<scf::ForOp>(loc, original.getLowerBound(),
      add(b, loc, original.getLowerBound(), span), step, initial);
  OpBuilder invariantBuilder(vectorLoop);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(vectorLoop.getBody());
    SmallVector<Value> results;
    for (int64_t part = 0; part < partitions; ++part) {
      Value coordinate = add(b, loc, vectorLoop.getInductionVar(), multiply(b, loc, span, index(b, loc, part)));
      auto inputs = vectorBody(coordinate, invariantBuilder);
      if (!inputs.empty()) {
        auto partial = horizontalReduce(b, loc, inputs, combineValues);
        auto accumulated = combineValues(
            vectorLoop.getRegionIterArgs().slice(part * combines.size(), combines.size()), partial, 0);
        llvm::append_range(results, accumulated);
      }
    }
    if (!results.empty()) b.create<scf::YieldOp>(loc, results);
  }
  SmallVector<Value> combined;
  for (auto [number, combine] : llvm::enumerate(combines)) {
    SmallVector<Value> leaves;
    for (int64_t part = 0; part < partitions; ++part)
      leaves.push_back(vectorLoop.getResult(part * combines.size() + number));
    while (leaves.size() > 1) {
      SmallVector<Value> level;
      for (unsigned part = 0; part < leaves.size(); part += 2)
        level.push_back(merge(combine, leaves[part], leaves[part + 1]));
      leaves = std::move(level);
    }
    combined.push_back(leaves.front());
  }
  original.setLowerBound(full);
  if (!reductionInputs.empty()) original.getInitArgsMutable().assign(combined);
  if (partitions > 1) vectorize(original, width, replicas);
  else if (replicas > 1) vectorize(original, width, 1);
  original->removeAttr("intent_cpu.reduction_order");
}

}

LogicalResult vectorizeLoop(scf::ForOp loop, int64_t width, int64_t replicas) {
  if (width <= 0 || replicas <= 0)
    return loop.emitError("CPU loop vectorization requires positive width and replicas");
  if (failed(foldLoopAccessSubviews(loop))) return failure();
  vectorize(loop, width, replicas);
  return success();
}

}
