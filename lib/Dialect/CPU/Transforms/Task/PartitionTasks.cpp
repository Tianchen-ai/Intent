#include "Intent/Dialect/CPU/Transforms/Task/Tasks.h"
#include "AtomicTaskOwnership.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <utility>

using namespace mlir;

namespace intent::cpu {
namespace {

std::pair<Value, Value> balancedTaskInterval(OpBuilder &b, Location loc,
                                            Value ordinal, Value width,
                                            Value remainder) {
  Value extraBefore = b.create<arith::MinSIOp>(loc, ordinal, remainder);
  Value begin = add(b, loc, multiply(b, loc, ordinal, width), extraBefore);
  Value extra = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ordinal, remainder);
  Value count = add(b, loc, width,
      b.create<arith::SelectOp>(loc, extra, index(b, loc, 1), index(b, loc, 0)));
  return {begin, add(b, loc, begin, count)};
}

Value worksetBoundary(OpBuilder &b, Location loc, Value ordinal,
                      Value chunk, Value fullChunks, Value tail) {
  Value full = b.create<arith::MinSIOp>(loc, ordinal, fullChunks);
  Value partial = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                       ordinal, fullChunks);
  return add(b, loc, multiply(b, loc, full, chunk),
      b.create<arith::SelectOp>(loc, partial, tail, index(b, loc, 0)));
}

SmallVector<scf::ParallelOp> partitionScalarSums(func::FuncOp function, int64_t grain) {
  SmallVector<scf::ParallelOp> partitioned;
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  if (capabilities.getWorkers() <= 1) return partitioned;
  int64_t limit = std::min(grain, capabilities.getPrivateBytes() / 4 / capabilities.getWorkers()) * capabilities.getWorkers();
  if (!limit) return partitioned;
  int64_t capacity = limit / grain + (limit % grain != 0);
  SmallVector<ReduceOp> reductions(function.front().getOps<ReduceOp>());
  for (ReduceOp operation : reductions) {
    if (!operation.getResult().getType().isF32() || !operation.getOrder().getAdjacentReassociation() ||
        !matchPattern(operation.getInitial(), m_PosZeroFloat())) continue;
    Block &body = operation.getCombine().front();
    auto combine = body.getTerminator()->getOperand(0).getDefiningOp<arith::AddFOp>();
    if (!combine || !body.getArgument(0).hasOneUse() ||
        !llvm::is_contained(combine->getOperands(), body.getArgument(0))) continue;
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    Value zero = index(b, loc, 0), one = index(b, loc, 1);
    Value count = b.create<arith::CeilDivSIOp>(loc, operation.getExtent(), index(b, loc, 4096));
    count = b.create<arith::MinSIOp>(loc, index(b, loc, limit),
        b.create<arith::MaxSIOp>(loc, one, count));
    Value chunk = index(b, loc, grain);
    Value taskCount = b.create<arith::CeilDivSIOp>(loc, count, chunk);
    Value width = b.create<arith::DivSIOp>(loc, operation.getExtent(), count);
    Value remainder = b.create<arith::RemSIOp>(loc, operation.getExtent(), count);
    Value partials = b.create<memref::AllocOp>(loc, MemRefType::get({capacity}, b.getF32Type()));
    auto tasks = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{taskCount}, ValueRange{one});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(tasks.getBody());
      Value group = tasks.getInductionVars()[0];
      // Consume the task grain before forming partial reductions. These are
      // exactly the adjacent intervals that the generic task partitioner would
      // assign to this task, now with one accumulator and one published value.
      Value first = multiply(b, loc, group, chunk);
      Value last = b.create<arith::MinSIOp>(loc, add(b, loc, first, chunk), count);
      auto boundary = [&](Value ordinal) {
        return add(b, loc, multiply(b, loc, ordinal, width),
                   b.create<arith::MinSIOp>(loc, ordinal, remainder));
      };
      Value begin = boundary(first);
      Value size = b.create<arith::SubIOp>(loc, boundary(last), begin);
      SmallVector<Value> inputs;
      for (auto [input, attr] : llvm::zip(operation.getInputs(), operation.getIndexingMaps())) {
        auto type = dyn_cast<MemRefType>(input.getType());
        if (!type) { inputs.push_back(input); continue; }
        SmallVector<OpFoldResult> offsets, sizes, strides(type.getRank(), b.getIndexAttr(1));
        for (AffineExpr expression : cast<AffineMapAttr>(attr).getValue().getResults()) {
          bool varying = isa<AffineDimExpr>(expression);
          offsets.push_back(varying ? OpFoldResult(begin) : OpFoldResult(b.getIndexAttr(0)));
          sizes.push_back(varying ? OpFoldResult(size) : OpFoldResult(b.getIndexAttr(1)));
        }
        inputs.push_back(b.create<memref::SubViewOp>(loc, input, offsets, sizes, strides));
      }
      auto partial = b.create<ReduceOp>(loc, operation.getResult().getType(), size,
          operation.getInitial(), inputs, operation.getIndexingMaps(), operation.getOrder());
      partial->setAttr("intent_cpu.implementation", operation->getAttr("intent_cpu.implementation"));
      IRMapping mapping;
      operation.getCombine().cloneInto(&partial.getCombine(), mapping);
      b.create<memref::StoreOp>(loc, partial.getResult(), partials, ValueRange{group});
    }
    auto result = b.create<ReduceOp>(loc, b.getF32Type(), taskCount, operation.getInitial(), ValueRange{partials},
        b.getArrayAttr({AffineMapAttr::get(b.getMultiDimIdentityMap(1))}), operation.getOrder());
    result->setAttr("intent_cpu.implementation", operation->getAttr("intent_cpu.implementation"));
    {
      OpBuilder::InsertionGuard guard(b);
      Block *merge = &result.getCombine().emplaceBlock();
      merge->addArguments(TypeRange{b.getF32Type(), b.getF32Type()}, {loc, loc});
      b.setInsertionPointToStart(merge);
      b.create<YieldOp>(loc, b.create<arith::AddFOp>(loc, merge->getArgument(0), merge->getArgument(1)).getResult());
    }
    b.create<memref::DeallocOp>(loc, partials);
    operation.replaceAllUsesWith(result.getResult());
    operation.erase();
    partitioned.push_back(tasks);
  }
  return partitioned;
}

LogicalResult exposeStructuredWorksets(func::FuncOp function,
    const ImplementationRegistry &implementations, ArrayRef<Value> leadingExtents,
    int64_t grain, SmallVectorImpl<scf::ParallelOp> &assigned) {
  SmallVector<linalg::GenericOp> computations(function.front().getOps<linalg::GenericOp>());
  for (auto operation : computations) {
    StorageAnalysis storage(function);
    if (!operation.getNumLoops() || operation.getNumResults() || operation.getOutputs().empty() ||
        operation.getIteratorTypesArray()[0] != utils::IteratorType::parallel) continue;
    if (!leadingExtents.empty() && isMatrixContraction(operation)) continue;
    auto maps = operation.getIndexingMapsArray();
    if (llvm::any_of(maps, [](AffineMap map) {
          return map.getNumSymbols() || llvm::any_of(map.getResults(), [](AffineExpr expression) {
            return !isa<AffineDimExpr, AffineConstantExpr>(expression);
          });
        }) || llvm::any_of(ArrayRef<AffineMap>(maps).drop_front(operation.getNumDpsInputs()),
                           [](AffineMap map) {
                             if (!map.getNumResults() || map.getResult(0) != getAffineDimExpr(0, map.getContext())) return true;
                             return llvm::any_of(map.getResults().drop_front(), [](AffineExpr expression) {
                               auto dimension = dyn_cast<AffineDimExpr>(expression);
                               return dimension && dimension.getPosition() == 0;
                             });
                           })) continue;
    bool independent = true;
    for (auto [number, destination] : llvm::enumerate(operation.getOutputs())) {
      for (Value other : operation.getOutputs().drop_front(number + 1))
        independent &= storage.disjoint(destination, other);
      for (auto [inputNumber, input] : llvm::enumerate(operation.getInputs())) {
        if (!isa<MemRefType>(input.getType())) continue;
        independent &= storage.disjoint(input, destination) ||
            (input == destination && maps[inputNumber] == maps[operation.getNumDpsInputs() + number]);
      }
    }
    for (Operation &body : operation.getRegion().front().without_terminator()) {
      if (isMemoryEffectFree(&body)) continue;
      auto load = dyn_cast<memref::LoadOp>(body);
      if (!load) { independent = false; break; }
      for (Value output : operation.getOutputs())
        independent &= storage.disjoint(load.getMemref(), output);
    }
    if (!independent) continue;
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    Value extent = b.createOrFold<memref::DimOp>(loc, operation.getOutputs()[0], 0);
    if (!leadingExtents.empty() && !llvm::any_of(leadingExtents, [&](Value candidate) {
          auto constant = getConstantIntValue(extent);
          return extent == candidate || (constant && constant == getConstantIntValue(candidate));
        })) continue;
    WorksetRows requirement;
    if (auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation")) {
      auto implementation = implementations.lookup(operation);
      if (failed(implementation)) return failure();
      if ((*implementation)->worksetRows)
        requirement = (*implementation)->worksetRows(operation, binding);
      if (requirement.unit <= 0 || requirement.replicas <= 0)
        return operation.emitError("implementation requires a positive leading parallel workset");
    }
    int64_t rows = requirement.unit;
    if (!leadingExtents.empty() && rows != 1) continue;
    Value zero = index(b, loc, 0), one = index(b, loc, 1);
    Value window = index(b, loc, rows);
    Value count = rows == 1 ? extent : b.create<arith::CeilDivSIOp>(loc, extent, window);
    scf::ParallelOp workset;
    Value row;
    OpFoldResult rowCount = b.getIndexAttr(1);
    if (grain > 0 && requirement.replicas > 1) {
      // Assign exactly the original width-sized work items and grain to workers
      // before combining adjacent items. A larger register group must not
      // decrease the task count or cross an existing worker boundary.
      auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>(
          "intent_cpu.capabilities");
      Value chunk = index(b, loc, grain);
      Value chunks = b.create<arith::CeilDivSIOp>(loc, count, chunk);
      auto active = b.create<scf::IfOp>(loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, chunks, zero), false);
      b.setInsertionPointToStart(active.thenBlock());
      Value workers = b.create<arith::MinSIOp>(loc, chunks,
          index(b, loc, capabilities.getWorkers()));
      Value width = b.create<arith::DivSIOp>(loc, chunks, workers);
      Value remainder = b.create<arith::RemSIOp>(loc, chunks, workers);
      Value fullChunks = b.create<arith::DivSIOp>(loc, count, chunk);
      Value tail = b.create<arith::RemSIOp>(loc, count, chunk);
      Value fullRows = b.create<arith::DivSIOp>(loc, extent, window);
      Value rowTail = b.create<arith::RemSIOp>(loc, extent, window);
      workset = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{workers}, ValueRange{one});
      assigned.push_back(workset);
      b.setInsertionPointToStart(workset.getBody());
      auto [first, last] = balancedTaskInterval(b, loc,
          workset.getInductionVars()[0], width, remainder);
      auto boundary = [&](Value ordinal) {
        Value item = worksetBoundary(b, loc, ordinal, chunk, fullChunks, tail);
        return worksetBoundary(b, loc, item, window, fullRows, rowTail);
      };
      row = boundary(first);
      rowCount = b.create<arith::SubIOp>(loc, boundary(last), row).getResult();
    } else {
      workset = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{count}, ValueRange{one});
      b.setInsertionPointToStart(workset.getBody());
      row = rows == 1 ? workset.getInductionVars()[0]
                     : multiply(b, loc, workset.getInductionVars()[0], window);
      if (rows != 1)
        rowCount = b.create<arith::MinSIOp>(loc, window,
            b.create<arith::SubIOp>(loc, extent, row)).getResult();
    }
    SmallVector<Value> inputs, outputs;
    for (auto [number, operand] : llvm::enumerate(operation->getOperands())) {
      Value selected = operand;
      if (auto type = dyn_cast<MemRefType>(operand.getType())) {
        SmallVector<OpFoldResult> offsets, sizes, strides(type.getRank(), b.getIndexAttr(1));
        for (auto [axis, expression] : llvm::enumerate(maps[number].getResults())) {
          auto dimension = dyn_cast<AffineDimExpr>(expression);
          bool leading = dimension && dimension.getPosition() == 0;
          offsets.push_back(leading ? OpFoldResult(row) : OpFoldResult(b.getIndexAttr(0)));
          sizes.push_back(leading ? rowCount
              : type.isDynamicDim(axis) ? OpFoldResult(b.create<memref::DimOp>(loc, operand, axis).getResult())
                                        : OpFoldResult(b.getIndexAttr(type.getDimSize(axis))));
        }
        selected = b.create<memref::SubViewOp>(loc, operand, offsets, sizes, strides);
      }
      (number < static_cast<size_t>(operation.getNumDpsInputs()) ? inputs : outputs).push_back(selected);
    }
    auto tile = b.create<linalg::GenericOp>(loc, inputs, outputs, maps, operation.getIteratorTypesArray(),
        [&](OpBuilder &nested, Location location, ValueRange arguments) {
          IRMapping mapping;
          Block &body = operation.getRegion().front();
          mapping.map(body.getArguments(), arguments);
          for (Operation &instruction : body.without_terminator()) {
            if (auto coordinate = dyn_cast<linalg::IndexOp>(instruction); coordinate && coordinate.getDim() == 0) {
              Value coordinateRow = row;
              if (rows != 1)
                coordinateRow = add(nested, location, row, nested.create<linalg::IndexOp>(location, 0));
              mapping.map(coordinate.getResult(), coordinateRow);
            } else nested.clone(instruction, mapping);
          }
          SmallVector<Value> results;
          for (Value value : body.getTerminator()->getOperands()) results.push_back(mapping.lookupOrDefault(value));
          nested.create<linalg::YieldOp>(location, results);
        });
    tile->setAttrs(operation->getAttrs());
    operation.erase();
  }
  return success();
}

}

LogicalResult exposeStructuredWorksets(func::FuncOp function,
    const ImplementationRegistry &implementations, ArrayRef<Value> leadingExtents) {
  SmallVector<scf::ParallelOp> assigned;
  return exposeStructuredWorksets(function, implementations, leadingExtents, 0, assigned);
}

namespace {

LogicalResult partition(scf::ParallelOp root, int64_t grain, bool distributeToWorkers) {
  // Rectangularity depends on invariant bounds, not where their producers were
  // first materialized. Native LICM keeps effects and zero-trip speculation safe.
  root.walk<WalkOrder::PostOrder>([](scf::ParallelOp parallel) {
    moveLoopInvariantCode(cast<LoopLikeOpInterface>(parallel.getOperation()));
  });
  auto function = root->getParentOfType<func::FuncOp>();
  DominanceInfo dominance(function);
  StorageAnalysis analysis(function);
  auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  auto rectangular = [&](scf::ParallelOp parallel) {
    for (auto [begin, end, step] : llvm::zip(parallel.getLowerBound(), parallel.getUpperBound(), parallel.getStep()))
      if (!matchPattern(begin, m_Zero()) || !matchPattern(step, m_One()) || !dominance.dominates(end, root))
        return false;
    return true;
  };
  auto duplicable = [&](Operation *operation) {
    if (operation->getNumRegions()) return false;
    if (isMemoryEffectFree(operation)) return true;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      auto view = analysis.externalView(load.getMemref());
      return view && view.getAccess() == 0 && interface.getDisjointOutputs();
    }
    return false;
  };
  SmallVector<scf::ParallelOp> nest;
  scf::ParallelOp current = root;
  while (current) {
    if (current.getNumResults()) return current.emitError("CPU task reduction is not implemented");
    nest.push_back(current);
    scf::ParallelOp child;
    bool flatten = true;
    for (Operation &operation : current.getBody()->without_terminator()) {
      if (auto parallel = dyn_cast<scf::ParallelOp>(&operation)) {
        if (child) { flatten = false; break; }
        child = parallel;
      }
    }
    if (child) {
      for (Operation &operation : current.getBody()->without_terminator())
        if (&operation != child && !duplicable(&operation)) flatten = false;
    }
    current = child && flatten && rectangular(child) ? child : scf::ParallelOp();
  }
  SmallVector<Value> extents, coordinates;
  for (scf::ParallelOp parallel : nest) {
    for (auto [begin, end, step, iv] : llvm::zip(parallel.getLowerBound(), parallel.getUpperBound(),
                                               parallel.getStep(), parallel.getInductionVars())) {
      if (!matchPattern(begin, m_Zero()) || !matchPattern(step, m_One()) ||
          !dominance.dominates(end, root))
        return parallel.emitError("CPU tasks require a rectangular zero-based unit-step workset");
      extents.push_back(end);
      coordinates.push_back(iv);
    }
  }
  OpBuilder b(root);
  Location loc = root.getLoc();
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  Value size = one;
  for (Value extent : extents) size = multiply(b, loc, size, extent);
  Value chunk = index(b, loc, grain);
  Value taskCount = b.create<arith::CeilDivSIOp>(loc, size, chunk);
  Value count = taskCount, width, remainder, fullChunks, tail;
  if (distributeToWorkers) {
    auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
    auto active = b.create<scf::IfOp>(loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, taskCount, zero), false);
    b.setInsertionPointToStart(active.thenBlock());
    count = b.create<arith::MinSIOp>(loc, taskCount, index(b, loc, capabilities.getWorkers()));
    width = b.create<arith::DivSIOp>(loc, taskCount, count);
    remainder = b.create<arith::RemSIOp>(loc, taskCount, count);
    fullChunks = b.create<arith::DivSIOp>(loc, size, chunk);
    tail = b.create<arith::RemSIOp>(loc, size, chunk);
  }
  auto tasks = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{count}, ValueRange{one});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(tasks.getBody());
    Value begin, end;
    if (distributeToWorkers) {
      auto [first, last] = balancedTaskInterval(b, loc, tasks.getInductionVars()[0], width, remainder);
      // Preserve the original grain assignment, but decode its contiguous
      // logical interval only once. The rounded final chunk may exceed index
      // range, so materialize endpoints through the bounded full part and tail.
      auto boundary = [&](Value ordinal) {
        return worksetBoundary(b, loc, ordinal, chunk, fullChunks, tail);
      };
      begin = boundary(first);
      end = boundary(last);
    } else {
      begin = multiply(b, loc, tasks.getInductionVars()[0], chunk);
      end = b.create<arith::MinSIOp>(loc, add(b, loc, begin, chunk), size);
    }
    auto clonePoint = [&](ValueRange point) {
      IRMapping mapping;
      mapping.map(ValueRange(coordinates), point);
      std::function<void(scf::ParallelOp)> cloneBody = [&](scf::ParallelOp parallel) {
        for (Operation &operation : parallel.getBody()->without_terminator()) {
          if (auto child = dyn_cast<scf::ParallelOp>(&operation); child && llvm::is_contained(nest, child)) cloneBody(child);
          else b.clone(operation, mapping);
        }
      };
      cloneBody(root);
    };
    if (extents.size() == 1) {
      loop(b, loc, begin, end, 1, [&](Value linear) { clonePoint(ValueRange{linear}); });
    } else {
      Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, begin, end);
      auto active = b.create<scf::IfOp>(loc, nonempty, false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(active.thenBlock());
      auto decode = [&](Value ordinal) {
        SmallVector<Value> point(extents.size());
        for (int64_t axis = extents.size() - 1; axis >= 0; --axis) {
          point[axis] = axis == 0 ? ordinal
              : Value(b.create<arith::RemSIOp>(loc, ordinal, extents[axis]));
          if (axis) ordinal = b.create<arith::DivSIOp>(loc, ordinal, extents[axis]);
        }
        return point;
      };
      auto first = decode(begin);
      auto last = decode(b.create<arith::SubIOp>(loc, end, one));
      SmallVector<Value> lastEnds, point(extents.size());
      for (Value coordinate : last) lastEnds.push_back(add(b, loc, coordinate, one));
      Value boundary = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
      std::function<void(unsigned, Value, Value)> traverse = [&](unsigned axis, Value firstPrefix, Value lastPrefix) {
        Value lower = b.createOrFold<arith::SelectOp>(loc, firstPrefix, first[axis], zero);
        Value upper = b.createOrFold<arith::SelectOp>(loc, lastPrefix, lastEnds[axis], extents[axis]);
        loop(b, loc, lower, upper, 1, [&](Value coordinate) {
          point[axis] = coordinate;
          if (axis + 1 == extents.size()) { clonePoint(point); return; }
          Value atFirst = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, coordinate, first[axis]);
          Value atLast = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, coordinate, last[axis]);
          traverse(axis + 1, b.createOrFold<arith::AndIOp>(loc, firstPrefix, atFirst),
              b.createOrFold<arith::AndIOp>(loc, lastPrefix, atLast));
        });
      };
      traverse(0, boundary, boundary);
    }
  }
  SmallVector<scf::ParallelOp> nested;
  tasks.walk<WalkOrder::PostOrder>([&](scf::ParallelOp parallel) {
    if (parallel != tasks) nested.push_back(parallel);
  });
  for (scf::ParallelOp parallel : nested) {
    if (parallel.getNumResults()) return parallel.emitError("CPU task reduction is not implemented");
    OpBuilder nestedBuilder(parallel);
    IRMapping mapping;
    std::function<void(unsigned)> traverse = [&](unsigned axis) {
      if (axis == parallel.getNumLoops()) {
        for (Operation &operation : parallel.getBody()->without_terminator())
          nestedBuilder.clone(operation, mapping);
        return;
      }
      auto loop = nestedBuilder.create<scf::ForOp>(parallel.getLoc(), parallel.getLowerBound()[axis],
          parallel.getUpperBound()[axis], parallel.getStep()[axis]);
      OpBuilder::InsertionGuard guard(nestedBuilder);
      nestedBuilder.setInsertionPointToStart(loop.getBody());
      mapping.map(parallel.getInductionVars()[axis], loop.getInductionVar());
      traverse(axis + 1);
    };
    traverse(0);
    parallel.erase();
  }
  root.erase();
  return success();
}

void foldDisjointCompareExchange(func::FuncOp function, scf::ParallelOp root) {
  if (root->getBlock() != &function.front() || root.getNumLoops() != 1 || root.getNumResults()) return;
  auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  if (!interface || !interface.getDisjointOutputs()) return;
  SmallVector<AtomicCompareExchangeOp> exchanges;
  root.walk([&](AtomicCompareExchangeOp operation) { exchanges.push_back(operation); });
  if (exchanges.size() != 1) return;
  auto exchange = exchanges.front();
  auto element = dyn_cast<IntegerType>(exchange.getOldValue().getType());
  if (exchange.getOrdering() != AtomicOrdering::Relaxed || !element || !element.isSignless()) return;
  StorageAnalysis physical(function);
  Value point = root.getInductionVars()[0];
  auto projection = [&](Value memory, ValueRange indices) -> Value {
    while (auto cast = memory.getDefiningOp<memref::CastOp>()) memory = cast.getSource();
    if (auto view = memory.getDefiningOp<memref::SubViewOp>()) {
      if (view.getSourceType().getRank() != 1 ||
          view.getMixedOffsets()[0] != OpFoldResult(point) ||
          getConstantIntValue(view.getMixedSizes()[0]) != 1 ||
          getConstantIntValue(view.getMixedStrides()[0]) != 1 ||
          !(indices.empty() || (indices.size() == 1 && matchPattern(indices[0], m_Zero())))) return {};
      memory = view.getSource();
    } else if (indices.size() != 1 || indices[0] != point) return {};
    auto type = cast<MemRefType>(memory.getType());
    auto external = physical.externalView(memory);
    if (type.getRank() != 1 || !type.getLayout().isIdentity() ||
        memory != physical.uniqueOrigin(memory) || !external || external.getAccess() == 0) return {};
    return memory;
  };
  Value target = projection(exchange.getTarget(), exchange.getIndices());
  if (!target || physical.externalView(target).getAccess() != 2) return;
  SmallVector<std::pair<memref::LoadOp, Value>> loads;
  SmallVector<std::pair<memref::StoreOp, Value>> stores;
  SmallVector<Operation *> aliases;
  bool valid = true;
  root.walk([&](Operation *operation) {
    if (operation == root || operation == exchange || isa<scf::ReduceOp>(operation)) return;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      if (physical.isReadOnly(load.getMemref())) return;
      if (Value memory = projection(load.getMemref(), load.getIndices())) loads.emplace_back(load, memory);
      else valid = false;
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      if (Value memory = projection(store.getMemref(), store.getIndices())) stores.emplace_back(store, memory);
      else valid = false;
      return;
    }
    if (isa<memref::SubViewOp, memref::CastOp>(operation)) { aliases.push_back(operation); return; }
    if (isa<memref::DimOp>(operation)) return;
    if (!isMemoryEffectFree(operation) || operation->getNumRegions() ||
        llvm::any_of(operation->getOperandTypes(), [](Type type) { return isa<MemRefType>(type); }) ||
        llvm::any_of(operation->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) valid = false;
  });
  if (!valid) return;

  // Every writable address belongs to this point, and readonly ABI inputs
  // cannot overlap it. No other participant can observe the relaxed CAS.
  OpBuilder b(exchange);
  Location loc = exchange.getLoc();
  Value old = b.create<memref::LoadOp>(loc, target, ValueRange{point});
  Value success = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, old, exchange.getExpected());
  Value value = b.create<arith::SelectOp>(loc, success, exchange.getDesired(), old);
  b.create<memref::StoreOp>(loc, value, target, ValueRange{point});
  exchange.getOldValue().replaceAllUsesWith(old);
  exchange.getSuccess().replaceAllUsesWith(success);
  exchange.erase();
  for (auto [load, memory] : loads) {
    b.setInsertionPoint(load);
    load.replaceAllUsesWith(b.create<memref::LoadOp>(load.getLoc(), memory, ValueRange{point}).getResult());
    load.erase();
  }
  for (auto [store, memory] : stores) {
    b.setInsertionPoint(store);
    b.create<memref::StoreOp>(store.getLoc(), store.getValue(), memory, ValueRange{point});
    store.erase();
  }
  for (Operation *alias : llvm::reverse(aliases))
    if (alias->use_empty()) alias->erase();
}

void foldSequentialAtomicAdds(func::FuncOp function) {
  SmallVector<AtomicRMWOp> updates;
  function.walk([&](AtomicRMWOp update) {
    if (update.getOrdering() != AtomicOrdering::Relaxed || update.getKind() != AtomicRMWKind::Add ||
        !isa<FloatType>(update.getValue().getType())) return;
    // CPU task dispatches join before continuation; only sequential control
    // may enclose an update with no concurrent invocation-local participant.
    for (Operation *parent = update->getParentOp(); parent != function; parent = parent->getParentOp())
      if (!isa<scf::ForOp, scf::IfOp>(parent)) return;
    updates.push_back(update);
  });
  for (auto update : updates) {
    OpBuilder b(update);
    Location loc = update.getLoc();
    Value old = b.create<memref::LoadOp>(loc, update.getTarget(), update.getIndices());
    Value sum = b.create<arith::AddFOp>(loc, old, update.getValue());
    b.create<memref::StoreOp>(loc, sum, update.getTarget(), update.getIndices());
    update.getOldValue().replaceAllUsesWith(old);
    update.erase();
  }
}

}

LogicalResult partitionWorkset(scf::ParallelOp workset, int64_t grain) {
  if (grain <= 0)
    return workset.emitError("CPU workset partitioning requires a positive grain");
  if (workset->getParentOfType<scf::ParallelOp>() ||
      workset->getParentOfType<TasksOp>() || workset->getParentOfType<TaskDispatchOp>())
    return workset.emitError("CPU workset partitioning requires an independent root outside an existing task");
  return partition(workset, grain, true);
}

LogicalResult partitionTasks(func::FuncOp function, int64_t grain, const ImplementationRegistry &implementations) {
  auto reductions = partitionScalarSums(function, grain);
  SmallVector<scf::ParallelOp> assigned;
  if (failed(exposeStructuredWorksets(function, implementations, {}, grain, assigned)))
    return failure();
  SmallVector<scf::ParallelOp> roots;
  function.walk([&](scf::ParallelOp operation) {
    if (!operation->getParentOfType<scf::ParallelOp>()) roots.push_back(operation);
  });
  for (scf::ParallelOp root : roots) {
    // These worksets already contain one complete partial per final task.
    if (llvm::is_contained(reductions, root) || llvm::is_contained(assigned, root)) continue;
    foldDisjointCompareExchange(function, root);
    auto owners = partitionAtomicWorkset(function, root, grain);
    // Atomic row worksets have already consumed the grain and worker budget.
    if (owners) {
      if (failed(partition(owners, 1, false))) return failure();
    } else if (failed(partitionWorkset(root, grain))) return failure();
  }
  foldSequentialAtomicAdds(function);
  return success();
}

LogicalResult isolateTasks(func::FuncOp function) {
  SmallVector<QuantizeOp> preparations;
  function.walk([&](QuantizeOp operation) {
    if (!operation->getParentOfType<scf::ParallelOp>() &&
        !operation->getParentOfType<TasksOp>() &&
        !operation->getParentOfType<TaskDispatchOp>())
      preparations.push_back(operation);
  });
  for (auto preparation : preparations) {
    OpBuilder b(preparation);
    auto loc = preparation.getLoc();
    auto task = b.create<scf::ParallelOp>(loc, ValueRange{index(b, loc, 0)},
        ValueRange{index(b, loc, 1)}, ValueRange{index(b, loc, 1)});
    preparation->moveBefore(task.getBody()->getTerminator());
  }
  SmallVector<scf::ParallelOp> operations;
  function.walk([&](scf::ParallelOp operation) { operations.push_back(operation); });
  for (auto parallel : operations) {
    if (parallel.getNumLoops() != 1 || parallel.getNumResults() ||
        !matchPattern(parallel.getLowerBound()[0], m_Zero()) ||
        !matchPattern(parallel.getStep()[0], m_One()))
      return parallel.emitError("task isolation requires the partitioned one-dimensional workset");
    llvm::SetVector<Value> used;
    getUsedValuesDefinedAbove(parallel.getRegion(), used);
    SmallVector<Value> captures;
    for (Value value : used)
      if (!value.getDefiningOp<arith::ConstantOp>()) captures.push_back(value);
    OpBuilder b(parallel);
    auto tasks = b.create<TasksOp>(parallel.getLoc(), parallel.getUpperBound()[0], captures);
    Block *body = new Block;
    tasks.getBody().push_back(body);
    body->addArgument(b.getIndexType(), parallel.getLoc());
    IRMapping mapping;
    mapping.map(parallel.getInductionVars()[0], body->getArgument(0));
    for (Value capture : captures)
      mapping.map(capture, body->addArgument(capture.getType(), capture.getLoc()));
    b.setInsertionPointToStart(body);
    for (Value value : used)
      if (auto constant = value.getDefiningOp<arith::ConstantOp>()) b.clone(*constant, mapping);
    for (Operation &operation : parallel.getBody()->without_terminator()) b.clone(operation, mapping);
    b.create<TaskYieldOp>(parallel.getLoc());
    parallel.erase();
  }
  return success();
}

LogicalResult materializeTaskDispatches(func::FuncOp function) {
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  SmallVector<TasksOp> operations;
  function.walk([&](TasksOp tasks) { operations.push_back(tasks); });
  for (auto tasks : operations) {
    OpBuilder b(tasks);
    Location loc = tasks.getLoc();
    Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, tasks.getCount(), index(b, loc, 0));
    auto active = b.create<scf::IfOp>(loc, nonempty, false);
    b.setInsertionPointToStart(active.thenBlock());
    Value workers = b.create<arith::MinSIOp>(loc, tasks.getCount(), index(b, loc, capabilities.getWorkers()));
    auto dispatch = b.create<TaskDispatchOp>(loc, tasks.getCount(), workers);
    Block *body = &dispatch.getBody().emplaceBlock();
    body->addArgument(b.getIndexType(), loc);
    IRMapping mapping;
    Block &original = tasks.getBody().front();
    for (auto [argument, capture] : llvm::zip(original.getArguments().drop_front(), tasks.getCaptures()))
      mapping.map(argument, capture);
    b.setInsertionPointToStart(body);
    mapping.map(original.getArgument(0), body->getArgument(0));
    for (Operation &operation : original.without_terminator()) b.clone(operation, mapping);
    b.create<TaskYieldOp>(loc);
    tasks.erase();
  }
  return success();
}

void localizeTaskConstants(func::FuncOp function) {
  SmallVector<TaskDispatchOp> dispatches;
  function.walk<WalkOrder::PreOrder>([&](TaskDispatchOp dispatch) {
    dispatches.push_back(dispatch);
  });
  // Outer scopes are processed first so nested callbacks also receive their
  // own constants, rather than capturing the enclosing callback's values.
  for (TaskDispatchOp dispatch : dispatches) {
    llvm::SetVector<Value> used;
    getUsedValuesDefinedAbove(dispatch.getBody(), used);
    OpBuilder builder(&dispatch.getBody().front(),
                      dispatch.getBody().front().begin());
    for (Value value : used) {
      auto constant = value.getDefiningOp<arith::ConstantOp>();
      if (!constant) continue;
      Value local = builder.clone(*constant)->getResult(0);
      replaceAllUsesInRegionWith(value, local, dispatch.getBody());
    }
  }
}

}
