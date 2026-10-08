#include "Intent/Dialect/CPU/Transforms/Structure/ParallelReductions.h"
#include "ReductionSources.h"
#include "../Vector/ContiguousMemory.h"
#include "../Vector/ProducerVectorization.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <functional>
#include <limits>

using namespace mlir;

namespace intent::cpu {
namespace {

using Dimension = std::pair<Value, unsigned>;

SmallVector<Dimension> loopDimensions(linalg::GenericOp operation) {
  AffineMap shapeToLoops = operation.getShapesToLoopsMap();
  if (!shapeToLoops) return {};
  SmallVector<Dimension> dimensions, result;
  for (Value operand : operation->getOperands()) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type) continue;
    for (unsigned axis = 0; axis < type.getRank(); ++axis)
      dimensions.emplace_back(operand, axis);
  }
  for (AffineExpr expression : shapeToLoops.getResults()) {
    auto axis = dyn_cast<AffineDimExpr>(expression);
    if (!axis || axis.getPosition() >= dimensions.size()) return {};
    result.push_back(dimensions[axis.getPosition()]);
  }
  return result;
}

bool numeric(Type type) {
  return ProducerVectorization::isElementType(type);
}

} // namespace

std::optional<ParallelReduction>
queryParallelReduction(linalg::GenericOp operation) {
  if (operation.getNumResults() || operation.getOutputs().empty() ||
      !operation->hasAttr("intent_cpu.reduction_order")) return std::nullopt;
  ParallelReduction result;
  SmallVector<unsigned> free;
  for (auto [axis, iterator] : llvm::enumerate(operation.getIteratorTypesArray())) {
    if (iterator == utils::IteratorType::parallel) free.push_back(axis);
    else if (iterator == utils::IteratorType::reduction)
      result.reductionAxes.push_back(axis);
    else return std::nullopt;
  }
  if (free.size() != 1 || result.reductionAxes.empty()) return std::nullopt;
  result.freeAxis = free.front();
  auto maps = operation.getIndexingMapsArray();
  if (llvm::any_of(maps, [](AffineMap map) {
        return map.getNumSymbols() || !map.isProjectedPermutation(true);
      })) return std::nullopt;
  auto dimensions = loopDimensions(operation);
  if (dimensions.size() != operation.getNumLoops()) return std::nullopt;
  auto [bound, boundAxis] = dimensions[result.freeAxis];
  Block &body = operation.getRegion().front();
  unsigned inputs = operation.getNumDpsInputs();
  for (auto [number, output] : llvm::enumerate(operation.getOutputs())) {
    auto type = dyn_cast<MemRefType>(output.getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    AffineMap map = maps[inputs + number];
    if (!type || type.getRank() != 1 || !numeric(type.getElementType()) ||
        type.getElementType().isInteger(1) ||
        failed(type.getStridesAndOffset(strides, offset)) || strides[0] != 1 ||
        map.getNumResults() != 1 ||
        map.getResult(0) != getAffineDimExpr(result.freeAxis, operation.getContext()) ||
        body.getArgument(inputs + number).use_empty() ||
        !haveEqualExtents(ValueBoundsConstraintSet::Variable(bound, boundAxis),
                          ValueBoundsConstraintSet::Variable(output, 0)))
      return std::nullopt;
  }
  for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type) {
      if (!numeric(input.getType()) || maps[number].getNumResults()) return std::nullopt;
      continue;
    }
    if (!numeric(type.getElementType())) return std::nullopt;
    for (auto [axis, expression] : llvm::enumerate(maps[number].getResults())) {
      if (expression != getAffineDimExpr(result.freeAxis, operation.getContext())) continue;
      SmallVector<int64_t> strides;
      int64_t offset;
      if (static_cast<int64_t>(axis) + 1 != type.getRank() || type.getElementType().isInteger(1) ||
          failed(type.getStridesAndOffset(strides, offset))) return std::nullopt;
      if (ShapedType::isDynamic(strides.back())) {
        if (!llvm::is_contained(result.guardedMemories, input))
          result.guardedMemories.push_back(input);
      } else if (strides.back() != 1) return std::nullopt;
    }
  }
  for (Operation &instruction : body.without_terminator()) {
    if (instruction.getNumRegions() || instruction.getNumResults() != 1 ||
        !numeric(instruction.getResult(0).getType()) ||
        !llvm::all_of(instruction.getOperandTypes(), numeric) ||
        !isMemoryEffectFree(&instruction) ||
        (!isa<arith::ConstantOp>(instruction) &&
         !instruction.hasTrait<OpTrait::Elementwise>())) return std::nullopt;
  }
  StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
  for (auto [number, output] : llvm::enumerate(operation.getOutputs())) {
    for (Value input : operation.getInputs())
      if (isa<MemRefType>(input.getType()) && !storage.disjoint(input, output))
        return std::nullopt;
    for (Value previous : operation.getOutputs().take_front(number))
      if (!storage.disjoint(previous, output)) return std::nullopt;
  }
  return result;
}

LogicalResult orientParallelReductions(func::FuncOp function) {
  SmallVector<linalg::GenericOp> operations;
  function.walk([&](linalg::GenericOp operation) { operations.push_back(operation); });
  for (auto operation : operations) {
    auto reduction = queryParallelReduction(operation);
    if (!reduction || reduction->freeAxis == 0) continue;
    SmallVector<unsigned> permutation{reduction->freeAxis};
    llvm::append_range(permutation, reduction->reductionAxes);
    IRRewriter rewriter(operation.getContext());
    if (failed(linalg::interchangeGenericOp(rewriter, operation, permutation)))
      return operation.emitError("cannot orient a proved independent reduction workset");
  }
  return success();
}

WorksetRows parallelReductionWorkset(linalg::GenericOp operation,
                                    int64_t width, int64_t replicas) {
  if (width <= 0 || replicas <= 0 ||
      replicas > std::numeric_limits<int64_t>::max() / width)
    return {0, 0};
  if (width == 1 || replicas == 1) return {width, 1};
  auto capabilities = operation->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>(
      "intent_cpu.capabilities");
  int64_t stateBytes = 0;
  for (Value output : operation.getOutputs()) {
    Type element = cast<MemRefType>(output.getType()).getElementType();
    int64_t bytes = element.isIndex() ? 8 : (element.getIntOrFloatBitWidth() + 7) / 8;
    if (width > (std::numeric_limits<int64_t>::max() - stateBytes) / bytes)
      return {width, 1};
    stateBytes += width * bytes;
  }
  // This bounds simultaneous private state, not native register allocation.
  replicas = std::min(replicas,
      std::max<int64_t>(1, capabilities.getPrivateBytes() / stateBytes));
  if (auto extent = constantExtentUpperBound(
          ValueBoundsConstraintSet::Variable(operation.getOutputs().front(), 0)))
    replicas = std::min(replicas, std::max<int64_t>(1, *extent / width));
  return {width, replicas};
}

FailureOr<bool> materializeParallelReduction(linalg::GenericOp operation,
    int64_t width, int64_t replicas, OpBuilder::Listener *listener,
    Value completedSupply, ArrayRef<Operation *> supplyReaders) {
  if (width <= 1) return false;
  auto reduction = queryParallelReduction(operation);
  if (!reduction) return false;
  auto workset = parallelReductionWorkset(operation, width, replicas);
  if (!workset.unit)
    return operation.emitError("independent reduction grouping requires a representable positive width"), failure();
  replicas = workset.replicas;
  ReductionSources sources(operation, completedSupply, supplyReaders);
  OpBuilder b(operation, listener);
  Location loc = operation.getLoc();
  auto maps = operation.getIndexingMapsArray();
  Block &body = operation.getRegion().front();
  SmallVector<Value> sizes, position(operation.getNumLoops());
  for (auto [memory, axis] : loopDimensions(operation))
    sizes.push_back(b.createOrFold<memref::DimOp>(loc, memory, axis));
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  Value active;
  for (unsigned axis : reduction->reductionAxes) {
    Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, sizes[axis], zero);
    active = active ? b.create<arith::AndIOp>(loc, active, nonempty) : nonempty;
  }
  auto nonempty = b.create<scf::IfOp>(loc, active, false);
  b.setInsertionPointToStart(nonempty.thenBlock());
  SmallVector<Value> guarded;
  for (auto [number, input] : llvm::enumerate(operation.getInputs()))
    if (!sources.replays(number) && llvm::is_contained(reduction->guardedMemories, input))
      guarded.push_back(input);
  auto contiguous = materializeContiguousMemoryGuard(b, loc, guarded);
  auto emit = [&](bool contiguousInputs) -> LogicalResult {
    IRMapping supplied;
    if (contiguousInputs) contiguous.bind(b, loc, supplied);
    auto broadcast = [&](Value value, int64_t lanes) -> Value {
      return lanes ? b.create<vector::BroadcastOp>(loc,
          VectorType::get({lanes}, value.getType()), value) : value;
    };
    auto coordinates = [&](AffineMap map) {
      SmallVector<Value> indices;
      for (AffineExpr expression : map.getResults()) {
        if (auto dimension = dyn_cast<AffineDimExpr>(expression))
          indices.push_back(position[dimension.getPosition()]);
        else indices.push_back(index(b, loc, cast<AffineConstantExpr>(expression).getValue()));
      }
      return indices;
    };
    auto compute = [&](ValueRange carry, int64_t lanes) -> FailureOr<SmallVector<Value>> {
      sources.beginMember();
      IRMapping mapping;
      for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
        auto type = dyn_cast<MemRefType>(input.getType());
        Value value = input;
        if (type) {
          auto indices = coordinates(maps[number]);
          bool varying = llvm::is_contained(maps[number].getResults(),
              getAffineDimExpr(reduction->freeAxis, b.getContext()));
          int64_t sourceLanes = varying ? lanes : 0;
          if (sources.replays(number)) {
            auto member = sources.materialize(b, number, indices, sourceLanes);
            if (failed(member)) return failure();
            value = *member;
          } else if (sourceLanes) {
            auto vectorType = VectorType::get({sourceLanes}, type.getElementType());
            if (contiguousInputs || !llvm::is_contained(guarded, input)) {
              value = b.create<vector::LoadOp>(loc, vectorType,
                  supplied.lookupOrDefault(input), indices);
            } else {
              SmallVector<Value> elements;
              Value begin = indices.back();
              for (int64_t lane = 0; lane < sourceLanes; ++lane) {
                indices.back() = lane ? add(b, loc, begin, index(b, loc, lane)) : begin;
                elements.push_back(b.create<memref::LoadOp>(loc, input, indices));
              }
              value = b.create<vector::FromElementsOp>(loc, vectorType, elements);
            }
          } else value = b.create<memref::LoadOp>(loc, input, indices);
          if (!sourceLanes) value = broadcast(value, lanes);
        } else value = broadcast(value, lanes);
        mapping.map(body.getArgument(number), value);
      }
      mapping.map(body.getArguments().drop_front(operation.getNumDpsInputs()), carry);
      auto mapped = [&](Value value) {
        if (!mapping.contains(value)) mapping.map(value, broadcast(value, lanes));
        return mapping.lookup(value);
      };
      for (Operation &instruction : body.without_terminator()) {
        if (lanes && isa<arith::ConstantOp>(instruction)) {
          mapping.map(instruction.getResult(0), broadcast(b.clone(instruction)->getResult(0), lanes));
          continue;
        }
        for (Value operand : instruction.getOperands()) mapped(operand);
        Operation *cloned = b.clone(instruction, mapping);
        if (lanes) cloned->getResult(0).setType(
            VectorType::get({lanes}, instruction.getResult(0).getType()));
      }
      SmallVector<Value> result;
      for (Value value : body.getTerminator()->getOperands()) result.push_back(mapped(value));
      return result;
    };
    auto tile = [&](Value coordinate, int64_t lanes, int64_t groups) -> LogicalResult {
      SmallVector<Value> groupCoordinates;
      for (int64_t group = 0; group < groups; ++group)
        groupCoordinates.push_back(group ? add(b, loc, coordinate, index(b, loc, group * lanes))
                                         : coordinate);
      SmallVector<Value> initial;
      for (Value selected : groupCoordinates)
        for (Value output : operation.getOutputs()) {
          auto type = cast<MemRefType>(output.getType());
          Value value = lanes ? Value(b.create<vector::LoadOp>(loc,
              VectorType::get({lanes}, type.getElementType()), output, ValueRange{selected}))
              : Value(b.create<memref::LoadOp>(loc, output, ValueRange{selected}));
          initial.push_back(value);
        }
      unsigned components = operation.getOutputs().size();
      std::function<FailureOr<SmallVector<Value>>(unsigned, ValueRange)> reduce =
          [&](unsigned depth, ValueRange carry) -> FailureOr<SmallVector<Value>> {
        if (depth == reduction->reductionAxes.size()) {
          SmallVector<Value> next;
          for (auto [group, selected] : llvm::enumerate(groupCoordinates)) {
            position[reduction->freeAxis] = selected;
            auto values = compute(carry.slice(group * components, components), lanes);
            if (failed(values)) return failure();
            llvm::append_range(next, *values);
          }
          return next;
        }
        unsigned axis = reduction->reductionAxes[depth];
        auto loop = b.create<scf::ForOp>(loc, zero, sizes[axis], one, carry);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(loop.getBody());
          position[axis] = loop.getInductionVar();
          auto next = reduce(depth + 1, loop.getRegionIterArgs());
          if (failed(next)) return failure();
          b.create<scf::YieldOp>(loc, *next);
        }
        return SmallVector<Value>(loop.getResults());
      };
      auto final = reduce(0, initial);
      if (failed(final)) return failure();
      for (auto [group, selected] : llvm::enumerate(groupCoordinates))
        for (auto [component, output] : llvm::enumerate(operation.getOutputs())) {
          Value value = (*final)[group * components + component];
          if (lanes) b.create<vector::StoreOp>(loc, value, output, ValueRange{selected});
          else b.create<memref::StoreOp>(loc, value, output, ValueRange{selected});
        }
      return success();
    };
    Value extent = sizes[reduction->freeAxis], step = index(b, loc, width);
    Value end = b.create<arith::SubIOp>(loc, extent, b.create<arith::RemSIOp>(loc, extent, step));
    LogicalResult status = success();
    Value begin = zero;
    if (replicas > 1) {
      int64_t groupWidth = width * replicas;
      Value groupStep = index(b, loc, groupWidth);
      begin = b.create<arith::SubIOp>(loc, extent,
          b.create<arith::RemSIOp>(loc, extent, groupStep));
      loop(b, loc, zero, begin, groupWidth,
          [&](Value coordinate) { status = tile(coordinate, width, replicas); });
      if (failed(status)) return failure();
    }
    loop(b, loc, begin, end, width, [&](Value coordinate) { status = tile(coordinate, width, 1); });
    if (failed(status)) return failure();
    loop(b, loc, end, extent, 1, [&](Value coordinate) { status = tile(coordinate, 0, 1); });
    return status;
  };
  if (contiguous.condition) {
    auto choice = b.create<scf::IfOp>(loc, contiguous.condition, true);
    b.setInsertionPointToStart(choice.thenBlock());
    if (failed(emit(true))) return failure();
    b.setInsertionPointToStart(choice.elseBlock());
    if (failed(emit(false))) return failure();
  } else if (failed(emit(true))) return failure();
  operation.erase();
  sources.eraseUnusedProducers();
  return true;
}

} // namespace intent::cpu
