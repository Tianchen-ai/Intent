#include "Intent/Dialect/CPU/Transforms/Structure/Computations.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ParallelReductions.h"
#include "ReductionSources.h"
#include "ReductionSupply.h"
#include "../Implementation/InputCopies.h"
#include "../Vector/ContiguousMemory.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "Intent/Dialect/CPU/Transforms/Vector/VectorReductions.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <limits>

using namespace mlir;

namespace intent::cpu {
namespace {

class LoopImplementationListener final : public OpBuilder::Listener {
public:
  explicit LoopImplementationListener(ImplementationAttr binding)
      : binding(binding) {}

  void notifyOperationInserted(Operation *operation,
                               OpBuilder::InsertPoint) final {
    if (!binding) return;
    operation->walk([&](scf::ForOp loop) {
      if (!loop->hasAttr("intent_cpu.implementation"))
        loop->setAttr("intent_cpu.implementation", binding);
    });
  }

private:
  ImplementationAttr binding;
};

SmallVector<Value> coordinates(OpBuilder &b, Location loc, AffineMap map,
                               ValueRange domain) {
  SmallVector<Value> result;
  for (AffineExpr expression : map.getResults()) {
    if (auto axis = dyn_cast<AffineDimExpr>(expression)) result.push_back(domain[axis.getPosition()]);
    else result.push_back(index(b, loc, cast<AffineConstantExpr>(expression).getValue()));
  }
  return result;
}

bool supportedMap(AffineMap map) {
  return !map.getNumSymbols() && llvm::all_of(map.getResults(), [](AffineExpr expression) {
    return isa<AffineDimExpr, AffineConstantExpr>(expression);
  });
}

FailureOr<bool> materializeProductReduction(linalg::GenericOp operation, int64_t width,
                                 OpBuilder::Listener *listener, ReductionSources &sources) {
  auto order = operation->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  if (!order || !order.getAdjacentReassociation() || width <= 1)
    return false;
  if (width & (width - 1)) return false;
  unsigned components = operation.getOutputs().size(), inputs = operation.getInputs().size();
  // Multi-output generics retain the source/combine boundary established by
  // structured reduction construction; single-output producer fusion does not.
  if (components < 2 || inputs < components || !operation.getNumLoops()) return false;
  auto iterators = operation.getIteratorTypesArray();
  if (iterators.back() != utils::IteratorType::reduction) return false;
  auto maps = operation.getIndexingMapsArray();
  Value domainSource;
  for (Value input : operation.getInputs().take_front(components))
    if (isa<MemRefType>(input.getType())) { domainSource = input; break; }
  auto sourceType = domainSource ? dyn_cast<MemRefType>(domainSource.getType()) : MemRefType{};
  if (!sourceType || sourceType.getRank() != operation.getNumLoops()) return false;
  auto scalarType = [](Type type) { return isa<FloatType, IntegerType, IndexType>(type); };
  SmallVector<AffineExpr> freeAxes;
  SmallVector<int64_t> outputShape;
  for (unsigned axis = 0; axis + 1 < operation.getNumLoops(); ++axis)
    if (iterators[axis] == utils::IteratorType::parallel) {
      freeAxes.push_back(getAffineDimExpr(axis, operation.getContext()));
      outputShape.push_back(sourceType.getDimSize(axis));
    }
  auto outputMap = AffineMap::get(operation.getNumLoops(), 0, freeAxes, operation.getContext());
  SmallVector<bool> vectorLoads(components, false);
  SmallVector<Value> guardedMemories;
  for (unsigned component = 0; component < components; ++component) {
    auto source = dyn_cast<MemRefType>(operation.getInputs()[component].getType());
    auto output = dyn_cast<MemRefType>(operation.getOutputs()[component].getType());
    Type element = source ? source.getElementType() : operation.getInputs()[component].getType();
    SmallVector<int64_t> strides;
    int64_t offset;
    if (!output || element != output.getElementType() || !scalarType(element) ||
        maps[inputs + component] != outputMap ||
        output.getShape() != ArrayRef<int64_t>(outputShape))
      return false;
    if (source) {
      if (source.getShape() != sourceType.getShape() || !maps[component].isIdentity() ||
          failed(source.getStridesAndOffset(strides, offset))) return false;
      vectorLoads[component] = strides.back() == 1 && !source.getElementType().isInteger(1);
      if (!sources.replays(component) && !source.getElementType().isInteger(1) &&
          ShapedType::isDynamic(strides.back()))
        guardedMemories.push_back(operation.getInputs()[component]);
    } else if (maps[component].getNumResults()) return false;
  }
  for (unsigned capture = components; capture < inputs; ++capture)
    for (AffineExpr expression : maps[capture].getResults())
      if (auto dimension = dyn_cast<AffineDimExpr>(expression))
        if (dimension.getPosition() + 1 == operation.getNumLoops()) return false;
  Block &body = operation.getRegion().front();
  if (!llvm::all_of(body.getArgumentTypes(), scalarType) ||
      llvm::any_of(body.getArguments().drop_front(inputs), [](BlockArgument argument) {
        return argument.use_empty();
      })) return false;
  for (Operation &instruction : body.without_terminator())
    if (instruction.getNumRegions() || instruction.getNumResults() != 1 ||
        !scalarType(instruction.getResult(0).getType()) ||
        !llvm::all_of(instruction.getOperandTypes(), scalarType) ||
        !isMemoryEffectFree(&instruction) ||
        (!isa<arith::ConstantOp>(instruction) && !instruction.hasTrait<OpTrait::Elementwise>()))
      return false;
  auto function = operation->getParentOfType<func::FuncOp>();
  bool accumulatePartials = order.getElementPermutation();
  SmallVector<IndependentReduction> independent;
  if (accumulatePartials) {
    for (unsigned component = 0; component < components; ++component) {
      auto reduction = matchIndependentReduction(body.getArgument(inputs + component),
          body.getArgument(component), body.getTerminator()->getOperand(component));
      if (!reduction) { independent.clear(); break; }
      independent.push_back(*reduction);
    }
  }
  StorageAnalysis storage(function);
  for (auto [component, output] : llvm::enumerate(operation.getOutputs())) {
    for (Value input : operation.getInputs())
      if (isa<MemRefType>(input.getType()) && !storage.disjoint(input, output)) return false;
    for (Value previous : operation.getOutputs().take_front(component))
      if (!storage.disjoint(previous, output)) return false;
  }

  OpBuilder b(operation, listener);
  Location loc = operation.getLoc();
  auto contiguous = materializeContiguousMemoryGuard(b, loc, guardedMemories);
  IRMapping supplied;
  SmallVector<Value> sizes, position(sourceType.getRank());
  for (int64_t axis = 0; axis < sourceType.getRank(); ++axis)
    sizes.push_back(b.create<memref::DimOp>(loc, domainSource, axis));
  auto combine = [&](ValueRange left, ValueRange right, ValueRange captures, int64_t lanes) {
    IRMapping mapping;
    for (unsigned component = 0; component < components; ++component) {
      mapping.map(body.getArgument(component), right[component]);
      mapping.map(body.getArgument(inputs + component), left[component]);
    }
    auto broadcast = [&](Value value) -> Value {
      return lanes ? Value(b.create<vector::BroadcastOp>(loc, VectorType::get({lanes}, value.getType()), value)) : value;
    };
    for (auto [number, capture] : llvm::enumerate(captures))
      mapping.map(body.getArgument(components + number), broadcast(capture));
    auto mapped = [&](Value value) {
      if (!mapping.contains(value)) mapping.map(value, broadcast(value));
      return mapping.lookup(value);
    };
    for (Operation &instruction : body.without_terminator()) {
      if (lanes && isa<arith::ConstantOp>(instruction)) {
        mapping.map(instruction.getResult(0), broadcast(b.clone(instruction)->getResult(0)));
        continue;
      }
      for (Value operand : instruction.getOperands()) mapped(operand);
      Operation *cloned = b.clone(instruction, mapping);
      if (lanes) cloned->getResult(0).setType(VectorType::get({lanes}, instruction.getResult(0).getType()));
    }
    SmallVector<Value> result;
    for (Value value : body.getTerminator()->getOperands()) result.push_back(mapped(value));
    return result;
  };
  std::function<LogicalResult(unsigned)> traverse = [&](unsigned axis) -> LogicalResult {
    if (axis + 1 < sizes.size()) {
      LogicalResult status = success();
      loop(b, loc, index(b, loc, 0), sizes[axis], 1, [&](Value coordinate) {
        position[axis] = coordinate;
        status = traverse(axis + 1);
      });
      return status;
    }
    Value zero = index(b, loc, 0), one = index(b, loc, 1), extent = sizes.back();
    Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, extent, zero);
    auto active = b.create<scf::IfOp>(loc, nonempty, false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(active.thenBlock());
    position.back() = zero;
    SmallVector<Value> initials, captures;
    auto outputPosition = coordinates(b, loc, outputMap, position);
    for (Value output : operation.getOutputs())
      initials.push_back(b.create<memref::LoadOp>(loc, output, outputPosition));
    for (unsigned number = components; number < inputs; ++number) {
      Value value = operation.getInputs()[number];
      if (isa<MemRefType>(value.getType()))
        value = b.create<memref::LoadOp>(loc, value, coordinates(b, loc, maps[number], position));
      captures.push_back(value);
    }
    Value step = index(b, loc, width);
    Value completeEnd = b.create<arith::SubIOp>(loc, extent, b.create<arith::RemSIOp>(loc, extent, step));
    auto horizontal = [&](ValueRange partial) {
      return horizontalReduce(b, loc, partial, [&](ValueRange left, ValueRange right, int64_t lanes) {
        return combine(left, right, captures, lanes);
      }, independent);
    };
    auto loadBlock = [&](Value begin) -> FailureOr<SmallVector<Value>> {
      position.back() = begin;
      sources.beginMember();
      SmallVector<Value> partial;
      for (auto [component, source] : llvm::enumerate(operation.getInputs().take_front(components))) {
        if (!isa<MemRefType>(source.getType())) {
          partial.push_back(b.create<vector::BroadcastOp>(loc, VectorType::get({width}, source.getType()), source));
          continue;
        }
        if (sources.replays(component)) {
          auto member = sources.materialize(b, component, position, width);
          if (failed(member)) return failure();
          partial.push_back(*member);
          continue;
        }
        auto type = VectorType::get({width}, cast<MemRefType>(source.getType()).getElementType());
        Value memory = supplied.lookupOrDefault(source);
        if (vectorLoads[component] || memory != source) {
          partial.push_back(b.create<vector::LoadOp>(loc, type, memory, position));
          continue;
        }
        // Scalar lane reads retain the memref's element representation and
        // strides, including bool storage whose vector packing is target-specific.
        SmallVector<Value> lanes, coordinate(position);
        for (int64_t lane = 0; lane < width; ++lane) {
          coordinate.back() = b.create<arith::AddIOp>(loc, position.back(), index(b, loc, lane));
          lanes.push_back(b.create<memref::LoadOp>(loc, memory, coordinate));
        }
        partial.push_back(b.create<vector::FromElementsOp>(loc, type, lanes));
      }
      return partial;
    };
    SmallVector<Value> reduced;
    if (accumulatePartials) {
      // Seed from members, then include the original accumulator exactly once.
      // It need not be an identity after an enclosing reduction is blocked.
      Value hasBlock = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, completeEnd, zero);
      auto full = b.create<scf::IfOp>(loc, TypeRange(initials), hasBlock, true);
      {
        OpBuilder::InsertionGuard fullGuard(b);
        b.setInsertionPointToStart(full.thenBlock());
        auto seeds = loadBlock(zero);
        if (failed(seeds)) return failure();
        auto blocks = b.create<scf::ForOp>(loc, step, completeEnd, step, *seeds);
        {
          OpBuilder::InsertionGuard blockGuard(b);
          b.setInsertionPointToStart(blocks.getBody());
          auto partial = loadBlock(blocks.getInductionVar());
          if (failed(partial)) return failure();
          b.create<scf::YieldOp>(loc, combine(blocks.getRegionIterArgs(), *partial, captures, width));
        }
        auto partial = horizontal(SmallVector<Value>(blocks.getResults()));
        b.create<scf::YieldOp>(loc, combine(initials, partial, captures, 0));
        b.setInsertionPointToStart(full.elseBlock());
        b.create<scf::YieldOp>(loc, initials);
      }
      reduced.assign(full.getResults().begin(), full.getResults().end());
    } else {
      auto blocks = b.create<scf::ForOp>(loc, zero, completeEnd, step, initials);
      {
        OpBuilder::InsertionGuard blockGuard(b);
        b.setInsertionPointToStart(blocks.getBody());
        auto members = loadBlock(blocks.getInductionVar());
        if (failed(members)) return failure();
        auto partial = horizontal(*members);
        b.create<scf::YieldOp>(loc, combine(blocks.getRegionIterArgs(), partial, captures, 0));
      }
      reduced.assign(blocks.getResults().begin(), blocks.getResults().end());
    }
    auto tail = b.create<scf::ForOp>(loc, completeEnd, extent, one, reduced);
    {
      OpBuilder::InsertionGuard tailGuard(b);
      b.setInsertionPointToStart(tail.getBody());
      position.back() = tail.getInductionVar();
      sources.beginMember();
      SmallVector<Value> values;
      for (auto [component, source] : llvm::enumerate(operation.getInputs().take_front(components))) {
        if (sources.replays(component)) {
          auto member = sources.materialize(b, component, position);
          if (failed(member)) return failure();
          values.push_back(*member);
        } else values.push_back(isa<MemRefType>(source.getType())
            ? Value(b.create<memref::LoadOp>(loc, supplied.lookupOrDefault(source), position)) : source);
      }
      b.create<scf::YieldOp>(loc, combine(tail.getRegionIterArgs(), values, captures, 0));
    }
    for (auto [value, output] : llvm::zip(tail.getResults(), operation.getOutputs()))
      b.create<memref::StoreOp>(loc, value, output, outputPosition);
    return success();
  };
  if (contiguous.condition) {
    // Both realizations consume the same proved source graph. Moving cloned
    // generics into these branches would lose its original producer scope.
    auto dispatch = b.create<scf::IfOp>(loc, contiguous.condition, true);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(dispatch.thenBlock());
    contiguous.bind(b, loc, supplied);
    if (failed(traverse(0))) return failure();
    supplied.clear();
    b.setInsertionPointToStart(dispatch.elseBlock());
    if (failed(traverse(0))) return failure();
  } else if (failed(traverse(0))) return failure();
  operation.erase();
  sources.eraseUnusedProducers();
  return true;
}

void collapseProductReductionAxes(linalg::GenericOp operation) {
  auto maps = operation.getIndexingMapsArray();
  auto order = operation->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  if (order && order.getElementPermutation() && operation.getOutputs().size() > 1) {
    auto iterators = operation.getIteratorTypesArray();
    SmallVector<ReassociationIndices> groups;
    for (auto [axis, iterator] : llvm::enumerate(iterators)) {
      if (iterator == utils::IteratorType::reduction && axis &&
          iterators[axis - 1] == utils::IteratorType::reduction)
        groups.back().push_back(axis);
      else groups.push_back({static_cast<int64_t>(axis)});
    }
    if (groups.size() < iterators.size() &&
        linalg::areDimSequencesPreserved(maps, groups)) {
      IRRewriter rewriter(operation.getContext());
      rewriter.setInsertionPoint(operation);
      auto collapsed = linalg::collapseOpIterationDims(operation, groups, rewriter);
      if (succeeded(collapsed)) {
        auto replacement = cast<linalg::GenericOp>(collapsed->collapsedOp.getOperation());
        replacement->setDiscardableAttrs(llvm::to_vector(operation->getDiscardableAttrs()));
        operation.erase();
      }
    }
  }
}

LogicalResult materialize(linalg::GenericOp operation, int64_t width, int64_t replicas,
                          OpBuilder::Listener *listener, Value completedSupply = {},
                          ArrayRef<Operation *> supplyReaders = {}) {
  if (operation.getOutputs().empty() || operation.getNumResults())
    return operation.emitError("CPU structured materialization requires destination buffers");
  auto maps = operation.getIndexingMapsArray();
  if (!llvm::all_of(maps, supportedMap))
    return operation.emitError("CPU pointwise coordinate map has no implemented scalar projection");
  auto parallel = materializeParallelReduction(operation, width, replicas, listener,
                                               completedSupply, supplyReaders);
  if (failed(parallel)) return operation.emitError("CPU parallel reduction cannot be materialized");
  if (*parallel) return success();
  ReductionSources sources(operation, completedSupply, supplyReaders);
  auto product = materializeProductReduction(operation, width, listener, sources);
  if (failed(product)) return operation.emitError("CPU reduction source member cannot be materialized");
  if (*product) return success();
  OpBuilder b(operation, listener);
  Location loc = operation.getLoc();
  SmallVector<Value> sizes(operation.getNumLoops()), position;
  for (auto [memory, map] : llvm::zip(operation->getOperands(), maps)) {
    if (!isa<MemRefType>(memory.getType())) continue;
    for (auto [axis, expression] : llvm::enumerate(map.getResults()))
      if (auto dimension = dyn_cast<AffineDimExpr>(expression))
        if (!sizes[dimension.getPosition()])
          sizes[dimension.getPosition()] = b.create<memref::DimOp>(loc, memory, axis);
  }
  if (llvm::any_of(sizes, [](Value size) { return !size; }))
    return operation.emitError("CPU structured traversal has an unbound loop extent");
  Block &body = operation.getRegion().front();
  auto order = operation->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  auto iterators = operation.getIteratorTypesArray();
  unsigned firstReduction = 0;
  while (firstReduction < iterators.size() &&
         iterators[firstReduction] == utils::IteratorType::parallel) ++firstReduction;
  SmallVector<AffineExpr> freeAxes;
  for (unsigned axis = 0; axis < firstReduction; ++axis)
    freeAxes.push_back(b.getAffineDimExpr(axis));
  auto freeMap = AffineMap::get(sizes.size(), 0, freeAxes, b.getContext());
  bool carryReduction = order && operation.getOutputs().size() == 1 && !sizes.empty() &&
      iterators.back() == utils::IteratorType::reduction &&
      llvm::all_of(ArrayRef(iterators).drop_front(firstReduction), [](utils::IteratorType iterator) {
        return iterator == utils::IteratorType::reduction;
      }) && maps.back() == freeMap && !body.getArguments().back().use_empty() &&
      llvm::all_of(body.without_terminator(), [](Operation &instruction) {
        return !instruction.getNumRegions() && isMemoryEffectFree(&instruction);
      });
  if (carryReduction) {
    auto function = operation->getParentOfType<func::FuncOp>();
    StorageAnalysis storage(function);
    Value output = operation.getOutputs()[0];
    for (Value input : operation.getInputs()) {
      if (!isa<MemRefType>(input.getType())) continue;
      if (storage.disjoint(input, output)) continue;
      carryReduction = false;
      break;
    }
  }
  auto compute = [&](Value carry = {}) -> FailureOr<SmallVector<Value>> {
    sources.beginMember();
    IRMapping mapping;
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      Value value = input;
      if (isa<MemRefType>(input.getType())) {
        auto selected = coordinates(b, loc, maps[number], position);
        if (sources.replays(number)) {
          auto member = sources.materialize(b, number, selected);
          if (failed(member)) return failure();
          value = *member;
        } else value = b.create<memref::LoadOp>(loc, input, selected);
      }
      mapping.map(body.getArgument(number), value);
    }
    for (auto [number, output] : llvm::enumerate(operation.getOutputs())) {
      unsigned argument = operation.getInputs().size() + number;
      if (!body.getArgument(argument).use_empty())
        mapping.map(body.getArgument(argument), carry ? carry : Value(b.create<memref::LoadOp>(
            loc, output, coordinates(b, loc, maps[argument], position))));
    }
    for (Operation &nested : body.without_terminator()) {
      if (auto index = dyn_cast<linalg::IndexOp>(nested)) mapping.map(index.getResult(), position[index.getDim()]);
      else b.clone(nested, mapping);
    }
    SmallVector<Value> results;
    for (Value value : body.getTerminator()->getOperands()) results.push_back(mapping.lookupOrDefault(value));
    return results;
  };
  std::function<LogicalResult(unsigned)> visit = [&](unsigned axis) -> LogicalResult {
    if (carryReduction && axis == firstReduction) {
      Value zero = index(b, loc, 0), one = index(b, loc, 1);
      Value nonempty;
      for (Value size : ArrayRef(sizes).drop_front(axis)) {
        Value present = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, size, zero);
        nonempty = nonempty ? b.create<arith::AndIOp>(loc, nonempty, present) : present;
      }
      auto active = b.create<scf::IfOp>(loc, nonempty, false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(active.thenBlock());
      Value output = operation.getOutputs()[0];
      Value initial = b.create<memref::LoadOp>(loc, output, position);
      std::function<FailureOr<Value>(unsigned, Value)> reduce =
          [&](unsigned current, Value seed) -> FailureOr<Value> {
        auto reduction = b.create<scf::ForOp>(loc, zero, sizes[current], one, ValueRange{seed});
        reduction->setAttr("intent_cpu.reduction_order", order);
        OpBuilder::InsertionGuard loopGuard(b);
        b.setInsertionPointToStart(reduction.getBody());
        position.push_back(reduction.getInductionVar());
        Value result;
        if (current + 1 == sizes.size()) {
          auto values = compute(reduction.getRegionIterArgs()[0]);
          if (failed(values)) return failure();
          result = values->front();
        } else {
          auto nested = reduce(current + 1, reduction.getRegionIterArgs()[0]);
          if (failed(nested)) return failure();
          result = *nested;
        }
        b.create<scf::YieldOp>(loc, result);
        position.pop_back();
        return reduction.getResult(0);
      };
      auto result = reduce(axis, initial);
      if (failed(result)) return failure();
      b.create<memref::StoreOp>(loc, *result, output, position);
      return success();
    }
    if (axis != sizes.size()) {
      LogicalResult status = success();
      loop(b, loc, index(b, loc, 0), sizes[axis], 1, [&](Value i) {
        position.push_back(i); status = visit(axis + 1); position.pop_back();
      });
      return status;
    }
    auto results = compute();
    if (failed(results)) return failure();
    for (auto [number, output] : llvm::enumerate(operation.getOutputs()))
      b.create<memref::StoreOp>(loc, (*results)[number], output,
          coordinates(b, loc, maps[operation.getInputs().size() + number], position));
    return success();
  };
  if (failed(visit(0))) return operation.emitError("CPU computation source member cannot be materialized");
  operation.erase();
  sources.eraseUnusedProducers();
  return success();
}

LogicalResult materialize(ReduceOp operation, OpBuilder::Listener *listener) {
  OpBuilder b(operation, listener);
  Location loc = operation.getLoc();
  auto reduction = b.create<scf::ForOp>(loc, index(b, loc, 0), operation.getExtent(),
      index(b, loc, 1), ValueRange{operation.getInitial()});
  reduction->setAttr("intent_cpu.reduction_order", operation.getOrder());
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(reduction.getBody());
    IRMapping mapping;
    Block &body = operation.getCombine().front();
    mapping.map(body.getArgument(0), reduction.getRegionIterArgs()[0]);
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      Value value = input;
      if (isa<MemRefType>(input.getType()))
        value = b.create<memref::LoadOp>(loc, input,
            coordinates(b, loc, cast<AffineMapAttr>(operation.getIndexingMaps()[number]).getValue(),
                        ValueRange{reduction.getInductionVar()}));
      mapping.map(body.getArgument(number + 1), value);
    }
    for (Operation &nested : body.without_terminator()) b.clone(nested, mapping);
    b.create<scf::YieldOp>(loc, mapping.lookupOrDefault(cast<YieldOp>(body.getTerminator()).getValue()));
  }
  operation.getResult().replaceAllUsesWith(reduction.getResult(0));
  operation.erase();
  return success();
}

LogicalResult materialize(ScanOp operation, int64_t width,
                          OpBuilder::Listener *listener) {
  OpBuilder b(operation, listener);
  Location loc = operation.getLoc();
  auto type = cast<MemRefType>(operation.getSources()[0].getType());
  int64_t scanAxis = operation.getAxis();
  if (width > 1 && !supportsVectorScan(operation))
    return operation.emitError("selected vector scan requires an elementwise last-axis scan with supported strides");
  ContiguousMemoryGuard contiguous;
  if (width > 1) {
    SmallVector<Value> memories(operation.getSources());
    llvm::append_range(memories, operation.getOutputs());
    contiguous = materializeContiguousMemoryGuard(b, loc, memories);
  }
  if (contiguous.condition) {
    auto dispatch = b.create<scf::IfOp>(loc, contiguous.condition, true);
    b.setInsertionPointToStart(dispatch.thenBlock());
    IRMapping mapping;
    contiguous.bind(b, loc, mapping);
    auto vectorScan = cast<ScanOp>(b.clone(*operation, mapping));
    if (failed(materialize(vectorScan, width, listener))) return failure();
    operation->moveBefore(dispatch.elseBlock(), dispatch.elseBlock()->begin());
    return materialize(operation, 1, listener);
  }
  bool pairBlocks = width > 1 && width <= std::numeric_limits<int64_t>::max() / 2 &&
      (type.isDynamicDim(scanAxis) || type.getDimSize(scanAxis) / width >= 2);
  if (pairBlocks) {
    StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
    for (auto [number, output] : llvm::enumerate(operation.getOutputs())) {
      for (Value source : operation.getSources())
        pairBlocks &= storage.disjoint(source, output);
      for (Value previous : operation.getOutputs().take_front(number))
        pairBlocks &= storage.disjoint(previous, output);
    }
  }
  SmallVector<Value> sizes, position(type.getRank());
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    sizes.push_back(b.create<memref::DimOp>(loc, operation.getSources()[0], axis));
  auto combine = [&](ValueRange left, ValueRange right, bool vectorized) {
    Block &body = operation.getCombine().front();
    IRMapping mapping;
    unsigned component = 0;
    for (Value value : left) mapping.map(body.getArgument(component++), value);
    for (Value value : right) mapping.map(body.getArgument(component++), value);
    for (Value capture : operation.getCaptures()) {
      Value value = vectorized ? Value(b.create<vector::BroadcastOp>(loc, VectorType::get({width}, capture.getType()), capture)) : capture;
      mapping.map(body.getArgument(component++), value);
    }
    for (Operation &instruction : body.without_terminator()) {
      if (vectorized && isa<arith::ConstantOp>(instruction)) {
        Value scalar = b.clone(instruction)->getResult(0);
        mapping.map(instruction.getResult(0), b.create<vector::BroadcastOp>(loc, VectorType::get({width}, scalar.getType()), scalar));
      } else {
        Operation *cloned = b.clone(instruction, mapping);
        if (vectorized) cloned->getResult(0).setType(VectorType::get({width}, instruction.getResult(0).getType()));
      }
    }
    SmallVector<Value> result;
    for (Value value : body.getTerminator()->getOperands()) result.push_back(mapping.lookupOrDefault(value));
    return result;
  };
  auto broadcast = [&](ValueRange scalars) {
    SmallVector<Value> result;
    for (Value scalar : scalars) result.push_back(b.create<vector::BroadcastOp>(loc, VectorType::get({width}, scalar.getType()), scalar));
    return result;
  };
  auto shift = [&](ValueRange vectors, ValueRange leading, int64_t offset) {
    SmallVector<int64_t> lanes;
    for (int64_t lane = 0; lane < width; ++lane) lanes.push_back(lane < offset ? lane : width + lane - offset);
    SmallVector<Value> result;
    for (auto [value, initial] : llvm::zip(vectors, leading))
      result.push_back(b.create<vector::ShuffleOp>(loc, initial, value, lanes));
    return result;
  };
  std::function<void(int64_t)> traverse = [&](int64_t axis) {
    if (axis != type.getRank()) {
      if (axis == scanAxis) { traverse(axis + 1); return; }
      loop(b, loc, index(b, loc, 0), sizes[axis], 1, [&](Value coordinate) {
        position[axis] = coordinate;
        traverse(axis + 1);
      });
      return;
    }
    Value extent = sizes[operation.getAxis()];
    Value begin = index(b, loc, 0);
    SmallVector<Value> initials(operation.getInitials());
    if (width > 1) {
      Value step = index(b, loc, width);
      Value completeEnd = b.create<arith::SubIOp>(loc, extent, b.create<arith::RemSIOp>(loc, extent, step));
      SmallVector<int64_t> reverseLanes;
      for (int64_t lane = width - 1; lane >= 0; --lane) reverseLanes.push_back(lane);
      auto blockPosition = [&](Value ordinal) {
        Value coordinate = ordinal;
        if (operation.getReverse()) coordinate = b.create<arith::SubIOp>(loc,
            b.create<arith::SubIOp>(loc, extent, step), coordinate);
        position[scanAxis] = coordinate;
      };
      auto localPrefix = [&](Value ordinal) {
        blockPosition(ordinal);
        SmallVector<Value> prefix;
        for (Value source : operation.getSources()) {
          auto vectorType = VectorType::get({width}, cast<MemRefType>(source.getType()).getElementType());
          Value values = b.create<vector::LoadOp>(loc, vectorType, source, position);
          if (operation.getReverse()) values = b.create<vector::ShuffleOp>(loc, values, values, reverseLanes);
          prefix.push_back(values);
        }
        auto identity = broadcast(operation.getInitials());
        // Each stage combines adjacent ranges in the selected traversal order.
        for (int64_t offset = 1; offset < width; offset *= 2)
          prefix = combine(shift(prefix, identity, offset), prefix, true);
        return prefix;
      };
      auto last = [&](ValueRange prefix) {
        SmallVector<Value> result;
        for (Value value : prefix)
          result.push_back(b.create<vector::ExtractElementOp>(loc, value, index(b, loc, width - 1)));
        return result;
      };
      auto storePrefix = [&](Value ordinal, ValueRange prefix, ValueRange incoming) {
        blockPosition(ordinal);
        SmallVector<Value> output(prefix);
        if (!operation.getInclusive()) output = shift(prefix, incoming, 1);
        for (auto [value, destination] : llvm::zip(output, operation.getOutputs())) {
          if (operation.getReverse()) value = b.create<vector::ShuffleOp>(loc, value, value, reverseLanes);
          b.create<vector::StoreOp>(loc, value, destination, position);
        }
      };
      if (pairBlocks) {
        Value pairStep = index(b, loc, 2 * width);
        Value pairEnd = b.create<arith::SubIOp>(loc, extent,
            b.create<arith::RemSIOp>(loc, extent, pairStep));
        auto pairs = b.create<scf::ForOp>(loc, begin, pairEnd, pairStep, initials);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(pairs.getBody());
          Value first = pairs.getInductionVar(), second = add(b, loc, first, step);
          auto firstPrefix = localPrefix(first);
          auto secondPrefix = localPrefix(second);
          auto firstTotal = last(firstPrefix), secondTotal = last(secondPrefix);
          // The carried dependency uses only scalar summaries of this pair;
          // output-vector adjustment does not feed the next iteration.
          auto total = combine(firstTotal, secondTotal, false);
          auto next = combine(pairs.getRegionIterArgs(), total, false);
          auto following = combine(pairs.getRegionIterArgs(), firstTotal, false);
          auto incoming = broadcast(pairs.getRegionIterArgs());
          auto followingIncoming = broadcast(following);
          firstPrefix = combine(incoming, firstPrefix, true);
          secondPrefix = combine(followingIncoming, secondPrefix, true);
          storePrefix(first, firstPrefix, incoming);
          storePrefix(second, secondPrefix, followingIncoming);
          b.create<scf::YieldOp>(loc, next);
        }
        begin = pairEnd;
        initials.assign(pairs.getResults().begin(), pairs.getResults().end());
      }
      auto blocks = b.create<scf::ForOp>(loc, begin, completeEnd, step, initials);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(blocks.getBody());
        Value ordinal = blocks.getInductionVar();
        auto prefix = localPrefix(ordinal);
        auto incoming = broadcast(blocks.getRegionIterArgs());
        prefix = combine(incoming, prefix, true);
        storePrefix(ordinal, prefix, incoming);
        b.create<scf::YieldOp>(loc, last(prefix));
      }
      begin = completeEnd;
      initials.assign(blocks.getResults().begin(), blocks.getResults().end());
    }
    auto scan = b.create<scf::ForOp>(loc, begin, extent, index(b, loc, 1), initials);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(scan.getBody());
    Value coordinate = scan.getInductionVar();
    if (operation.getReverse()) coordinate = b.create<arith::SubIOp>(loc,
        b.create<arith::SubIOp>(loc, extent, index(b, loc, 1)), coordinate);
    position[operation.getAxis()] = coordinate;
    SmallVector<Value> inputs;
    for (Value source : operation.getSources())
      inputs.push_back(b.create<memref::LoadOp>(loc, source, position));
    auto next = combine(scan.getRegionIterArgs(), inputs, false);
    ValueRange output = operation.getInclusive() ? ValueRange(next) : ValueRange(scan.getRegionIterArgs());
    for (auto [value, destination] : llvm::zip(output, operation.getOutputs()))
      b.create<memref::StoreOp>(loc, value, destination, position);
    b.create<scf::YieldOp>(loc, next);
  };
  traverse(0);
  operation.erase();
  return success();
}

} // namespace

LogicalResult materializeStructuredComputation(Operation *operation,
                                              int64_t width,
                                              ImplementationAttr loopBinding) {
  if (width <= 0)
    return operation->emitError("structured materialization requires a positive vector width");
  LoopImplementationListener listener(loopBinding);
  if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
    if (width > 1 && !loopBinding)
      return operation->emitError("vector materialization requires a selected implementation binding");
    if (width >= 16 && loopBinding) {
      auto copied = tryMaterializeInputCopy(generic, width, &listener);
      if (failed(copied)) return failure();
      if (*copied) return success();
    }
    int64_t replicas = width > 1
        ? implementationParameter(loopBinding, "register_replicas") : 1;
    auto supplied = materializeReductionSupplyGroup(generic, width,
        [&](linalg::GenericOp reader, Value source, ArrayRef<Operation *> readers) {
          ReductionSources members(reader, source, readers);
          for (auto [number, input] : llvm::enumerate(reader.getInputs()))
            if (input == source && !members.replays(number)) return failure();
          return materialize(reader, width, replicas, &listener, source, readers);
        });
    if (failed(supplied)) return failure();
    if (*supplied) return success();
    return materialize(generic, width, replicas, &listener);
  }
  if (auto reduce = dyn_cast<ReduceOp>(operation))
    return materialize(reduce, &listener);
  if (auto scan = dyn_cast<ScanOp>(operation))
    return materialize(scan, width, &listener);
  return operation->emitError("operation has no CPU structured materialization");
}

LogicalResult materializeStructuredComputations(
    func::FuncOp function,
    llvm::function_ref<LogicalResult(Operation *)> materializeOperation) {
  // Keep source definitions structured until a tuple consumer has had a chance
  // to supply its real members directly. The combine remains a general function
  // of two states, including when a source field is a constant fill.
  SmallVector<linalg::FillOp> fills;
  function.walk([&](linalg::FillOp fill) { fills.push_back(fill); });
  for (auto fill : fills) {
    if (fill.getOutputs().size() != 1 || fill.getNumResults())
      return fill.emitError("CPU fill requires one physical buffer");
    OpBuilder builder(fill);
    int64_t rank = cast<MemRefType>(fill.getOutputs()[0].getType()).getRank();
    auto generic = builder.create<linalg::GenericOp>(fill.getLoc(), fill.getInputs(), fill.getOutputs(),
        SmallVector<AffineMap>{AffineMap::get(rank, 0, {}, builder.getContext()),
                              builder.getMultiDimIdentityMap(rank)},
        SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
        [](OpBuilder &nested, Location loc, ValueRange arguments) {
          nested.create<linalg::YieldOp>(loc, arguments[0]);
        });
    if (Attribute binding = fill->getAttr("intent_cpu.implementation"))
      generic->setAttr("intent_cpu.implementation", binding);
    fill.erase();
  }
  SmallVector<linalg::GenericOp> products;
  function.walk([&](linalg::GenericOp operation) {
    if (operation.getOutputs().size() > 1 && !operation.getNumResults())
      products.push_back(operation);
  });
  for (auto operation : products) {
    auto uniform = ReductionSources(operation).foldUniformMembers();
    if (failed(uniform)) return operation.emitError("CPU uniform source member cannot be materialized");
  }
  while (true) {
    linalg::GenericOp supplied;
    function.walk([&](linalg::GenericOp producer) {
      if (!hasReductionSupplyGroup(producer)) return WalkResult::advance();
      supplied = producer;
      return WalkResult::interrupt();
    });
    if (!supplied) break;
    if (failed(materializeOperation(supplied))) return failure();
  }
  products.clear();
  function.walk([&](linalg::GenericOp operation) {
    if (operation.getOutputs().size() > 1 && !operation.getNumResults()) products.push_back(operation);
  });
  for (auto operation : products) {
    bool supplied = ReductionSources(operation).hasReplays();
    if (supplied && failed(materializeOperation(operation))) return failure();
  }
  function.walk([&](linalg::GenericOp operation) { collapseProductReductionAxes(operation); });
  SmallVector<Operation *> views;
  function.walk([&](Operation *operation) {
    if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp>(operation)) views.push_back(operation);
  });
  for (Operation *operation : views) {
    auto axes = unitReshapeAxes(operation);
    auto collapse = dyn_cast<memref::CollapseShapeOp>(operation);
    if (!axes && !collapse) return operation->emitError("CPU shape view materialization requires unit-axis reassociation");
    OpBuilder b(operation);
    Location loc = operation->getLoc();
    auto type = cast<MemRefType>(operation->getResult(0).getType());
    SmallVector<int64_t> staticStrides;
    int64_t staticOffset;
    if (failed(type.getStridesAndOffset(staticStrides, staticOffset)))
      return operation->emitError("CPU shape view materialization requires strided storage");
    auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, operation->getOperand(0));
    SmallVector<OpFoldResult> sizes(type.getRank()), strides;
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      if (!type.isDynamicDim(axis)) sizes[axis] = b.getIndexAttr(type.getDimSize(axis));
      strides.push_back(b.getIndexAttr(ShapedType::isDynamic(staticStrides[axis]) ? 1 : staticStrides[axis]));
    }
    if (axes)
      for (auto [source, result] : *axes) {
        if (type.isDynamicDim(result)) sizes[result] = metadata.getSizes()[source];
        if (ShapedType::isDynamic(staticStrides[result])) strides[result] = metadata.getStrides()[source];
      }
    if (collapse)
      for (auto [result, group] : llvm::enumerate(collapse.getReassociationIndices())) {
        if (type.isDynamicDim(result)) {
          Value size = index(b, loc, 1);
          for (int64_t source : group)
            size = b.create<arith::MulIOp>(loc, size, metadata.getSizes()[source]);
          sizes[result] = size;
        }
        if (!axes && ShapedType::isDynamic(staticStrides[result])) {
          Value stride = metadata.getStrides()[group.back()];
          Value found = b.create<arith::ConstantIntOp>(loc, 0, 1);
          for (int64_t source : llvm::reverse(group)) {
            Value nonunit = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                metadata.getSizes()[source], index(b, loc, 1));
            Value choose = b.create<arith::AndIOp>(loc, nonunit,
                b.create<arith::XOrIOp>(loc, found, b.create<arith::ConstantIntOp>(loc, 1, 1)));
            stride = b.create<arith::SelectOp>(loc, choose, metadata.getStrides()[source], stride);
            found = b.create<arith::OrIOp>(loc, found, nonunit);
          }
          strides[result] = stride;
        }
      }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(operation))
      for (auto [source, group] : llvm::enumerate(expand.getReassociationIndices())) {
        Value stride = metadata.getStrides()[source];
        for (int64_t result : llvm::reverse(group)) {
          if (ShapedType::isDynamic(staticStrides[result])) strides[result] = stride;
          if (result != group.front() && type.getDimSize(result) != 1)
            stride = b.create<arith::MulIOp>(loc, stride, getValueOrCreateConstantIndexOp(b, loc, sizes[result]));
        }
      }
    OpFoldResult offset = ShapedType::isDynamic(staticOffset) ? OpFoldResult(metadata.getOffset())
                                                            : OpFoldResult(b.getIndexAttr(staticOffset));
    Value view = b.create<memref::ReinterpretCastOp>(loc, type, metadata.getBaseBuffer(), offset, sizes, strides);
    operation->getResult(0).replaceAllUsesWith(view);
    operation->erase();
  }
  SmallVector<Operation *> operations;
  function.walk([&](Operation *operation) {
    if (isa<linalg::GenericOp, ReduceOp, ScanOp>(operation)) operations.push_back(operation);
  });
  for (Operation *operation : operations)
    if (failed(materializeOperation(operation))) return failure();
  return success();
}

}
