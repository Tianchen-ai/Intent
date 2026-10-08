#include "Intent/Dialect/CPU/Transforms/Collective/Collectives.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>

using namespace mlir;

namespace intent::cpu {
namespace {

void copy(OpBuilder &builder, Location location, Value source, Value target) {
  if (isa<MemRefType>(source.getType()))
    builder.create<memref::CopyOp>(location, source, target);
  else if (cast<MemRefType>(target.getType()).getRank())
    builder.create<linalg::FillOp>(location, ValueRange{source}, ValueRange{target});
  else
    builder.create<memref::StoreOp>(location, source, target, ValueRange{});
}

SmallVector<Value> scratch(OpBuilder &builder, Location location,
                           ValueRange prototypes) {
  SmallVector<Value> result;
  for (Value value : prototypes) {
    auto type = collectiveStateType(value.getType());
    SmallVector<Value> sizes;
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      if (type.isDynamicDim(axis))
        sizes.push_back(builder.create<memref::DimOp>(location, value, axis));
    result.push_back(builder.create<memref::AllocOp>(location, type, sizes));
  }
  return result;
}

Value memberSlice(OpBuilder &builder, Location location, Value source,
                  ArrayRef<int64_t> axes, ValueRange coordinates) {
  auto type = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets, sizes, strides(type.getRank(), builder.getIndexAttr(1));
  SmallVector<int64_t> shape;
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    auto member = llvm::find(axes, axis);
    bool reduced = member != axes.end();
    offsets.push_back(reduced ? OpFoldResult(coordinates[member - axes.begin()])
                             : OpFoldResult(builder.getIndexAttr(0)));
    sizes.push_back(reduced ? OpFoldResult(builder.getIndexAttr(1))
        : type.isDynamicDim(axis)
            ? OpFoldResult(builder.create<memref::DimOp>(location, source, axis).getResult())
            : OpFoldResult(builder.getIndexAttr(type.getDimSize(axis))));
    if (!reduced) shape.push_back(type.getDimSize(axis));
  }
  auto result = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      shape, type, offsets, sizes, strides));
  return builder.create<memref::SubViewOp>(location, result, source, offsets, sizes, strides);
}

FailureOr<SmallVector<Value>> instantiateHelper(OpBuilder &builder, Region &region,
                                               ValueRange arguments) {
  Block &body = region.front();
  if (body.getNumArguments() != arguments.size())
    return region.getParentOp()->emitOpError("collective helper argument binding is incomplete"), failure();
  IRMapping mapping;
  for (auto [argument, input] : llvm::zip(body.getArguments(), arguments)) {
    if (input.getType() != argument.getType()) {
      if (!isa<MemRefType>(input.getType()) || !isa<MemRefType>(argument.getType()) ||
          !memref::CastOp::areCastCompatible(TypeRange{input.getType()}, TypeRange{argument.getType()}))
        return region.getParentOp()->emitOpError("collective helper argument has no compatible descriptor binding"), failure();
      input = builder.create<memref::CastOp>(region.getLoc(), argument.getType(), input);
    }
    mapping.map(argument, input);
  }
  for (Operation &nested : body.without_terminator())
    builder.clone(nested, mapping);
  SmallVector<Value> results;
  for (Value value : body.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  return results;
}

bool commonScalarDomain(SliceReduceOp operation) {
  Value first = operation.getSources().front();
  auto type = cast<MemRefType>(first.getType());
  for (Value source : operation.getSources().drop_front()) {
    if (cast<MemRefType>(source.getType()).getRank() != type.getRank()) return false;
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      if (!llvm::is_contained(operation.getAxes(), axis) &&
          !haveEqualExtents(ValueBoundsConstraintSet::Variable(first, axis),
                            ValueBoundsConstraintSet::Variable(source, axis)))
        return false;
  }
  return true;
}

bool formPointwiseReduction(SliceReduceOp operation) {
  if (llvm::any_of(operation.getCaptures(), [](Value value) {
        return !isa<IntegerType, IndexType, FloatType>(value.getType());
      })) return false;
  Block &body = operation.getCombine().front();
  auto scalar = [](Type type) { return isa<IntegerType, IndexType, FloatType>(type); };
  for (Operation &nested : body.without_terminator())
    if (nested.getNumRegions() || !isMemoryEffectFree(&nested) ||
        !llvm::all_of(nested.getOperandTypes(), scalar) ||
        !llvm::all_of(nested.getResultTypes(), scalar))
      return false;
  SmallVector<Value> reads(operation.getSources());
  llvm::append_range(reads, operation.getIdentities());
  llvm::append_range(reads, operation.getCaptures());
  StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
  for (auto [number, output] : llvm::enumerate(operation.getOutputs())) {
    for (Value input : reads)
      if (isa<MemRefType>(input.getType()) && !storage.disjoint(input, output))
        return false;
    for (Value previous : operation.getOutputs().take_front(number))
      if (!storage.disjoint(previous, output)) return false;
  }
  OpBuilder builder(operation);
  Location location = operation.getLoc();
  auto type = cast<MemRefType>(operation.getSources().front().getType());
  SmallVector<AffineExpr> freeAxes;
  SmallVector<utils::IteratorType> iterators;
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    bool reduced = llvm::is_contained(operation.getAxes(), axis);
    iterators.push_back(reduced ? utils::IteratorType::reduction : utils::IteratorType::parallel);
    if (!reduced) freeAxes.push_back(builder.getAffineDimExpr(axis));
  }
  for (auto [identity, output] : llvm::zip(operation.getIdentities(), operation.getOutputs()))
    copy(builder, location, identity, output);
  SmallVector<Value> inputs(operation.getSources());
  llvm::append_range(inputs, operation.getCaptures());
  SmallVector<AffineMap> maps(operation.getSources().size(), builder.getMultiDimIdentityMap(type.getRank()));
  maps.append(operation.getCaptures().size(), AffineMap::get(type.getRank(), 0, {}, builder.getContext()));
  maps.append(operation.getOutputs().size(), AffineMap::get(type.getRank(), 0, freeAxes, builder.getContext()));
  auto generic = builder.create<linalg::GenericOp>(location, inputs, operation.getOutputs(), maps, iterators,
      [&](OpBuilder &nested, Location loc, ValueRange arguments) {
        IRMapping mapping;
        unsigned count = operation.getSources().size();
        mapping.map(body.getArguments().take_front(count), arguments.drop_front(inputs.size()));
        mapping.map(body.getArguments().slice(count, count), arguments.take_front(count));
        mapping.map(body.getArguments().drop_front(2 * count), arguments.slice(count, operation.getCaptures().size()));
        for (Operation &instruction : body.without_terminator()) nested.clone(instruction, mapping);
        SmallVector<Value> yielded;
        for (Value value : body.getTerminator()->getOperands()) yielded.push_back(mapping.lookupOrDefault(value));
        nested.create<linalg::YieldOp>(loc, yielded);
      });
  generic->setAttr("intent_cpu.reduction_order", operation.getOrder());
  operation.erase();
  return true;
}

LogicalResult realize(SliceReduceOp operation) {
  bool destinationPassing = operation.isDestinationPassing();
  if (!destinationPassing) {
    if (!commonScalarDomain(operation))
      return operation.emitOpError("scalar combine components require a proven common free-axis domain; use complete slice identities for different component domains");
    if (formPointwiseReduction(operation)) return success();
  }
  OpBuilder builder(operation);
  Location location = operation.getLoc();
  auto states = scratch(builder, location, operation.getOutputs());
  auto next = scratch(builder, location, operation.getOutputs());
  for (auto [identity, state] : llvm::zip(operation.getIdentities(), states))
    copy(builder, location, identity, state);
  SmallVector<Value> reducedCoordinates;
  std::function<LogicalResult(unsigned)> reduce = [&](unsigned axis) -> LogicalResult {
    if (axis < operation.getAxes().size()) {
      Value extent = builder.create<memref::DimOp>(location, operation.getSources().front(), operation.getAxes()[axis]);
      auto loop = builder.create<scf::ForOp>(location, index(builder, location, 0), extent, index(builder, location, 1));
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      reducedCoordinates.push_back(loop.getInductionVar());
      LogicalResult status = reduce(axis + 1);
      reducedCoordinates.pop_back();
      return status;
    }
    if (destinationPassing) {
      SmallVector<Value> arguments(states);
      for (Value source : operation.getSources())
        arguments.push_back(memberSlice(builder, location, source, operation.getAxes(), reducedCoordinates));
      llvm::append_range(arguments, operation.getCaptures());
      llvm::append_range(arguments, next);
      if (failed(instantiateHelper(builder, operation.getCombine(), arguments))) return failure();
    } else {
      auto sourceType = cast<MemRefType>(operation.getSources().front().getType());
      SmallVector<Value> freeCoordinates;
      std::function<LogicalResult(int64_t)> visit = [&](int64_t axis) -> LogicalResult {
        if (axis < sourceType.getRank()) {
          if (llvm::is_contained(operation.getAxes(), axis)) return visit(axis + 1);
          Value extent = builder.create<memref::DimOp>(location, operation.getSources().front(), axis);
          auto loop = builder.create<scf::ForOp>(location, index(builder, location, 0), extent, index(builder, location, 1));
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(loop.getBody());
          freeCoordinates.push_back(loop.getInductionVar());
          LogicalResult status = visit(axis + 1);
          freeCoordinates.pop_back();
          return status;
        }
        SmallVector<Value> coordinates, arguments;
        unsigned free = 0;
        for (int64_t axis = 0; axis < sourceType.getRank(); ++axis) {
          auto reduced = llvm::find(operation.getAxes(), axis);
          coordinates.push_back(reduced == operation.getAxes().end() ? freeCoordinates[free++]
              : reducedCoordinates[reduced - operation.getAxes().begin()]);
        }
        for (Value state : states)
          arguments.push_back(builder.create<memref::LoadOp>(location, state, freeCoordinates));
        for (Value source : operation.getSources())
          arguments.push_back(builder.create<memref::LoadOp>(location, source, coordinates));
        llvm::append_range(arguments, operation.getCaptures());
        auto result = instantiateHelper(builder, operation.getCombine(), arguments);
        if (failed(result)) return failure();
        for (auto [value, destination] : llvm::zip(*result, next))
          builder.create<memref::StoreOp>(location, value, destination, freeCoordinates);
        return success();
      };
      if (failed(visit(0))) return failure();
    }
    for (auto [source, destination] : llvm::zip(next, states)) copy(builder, location, source, destination);
    return success();
  };
  if (failed(reduce(0))) return failure();
  for (auto [state, output] : llvm::zip(states, operation.getOutputs())) copy(builder, location, state, output);
  for (Value value : next) builder.create<memref::DeallocOp>(location, value);
  for (Value value : states) builder.create<memref::DeallocOp>(location, value);
  operation.erase();
  return success();
}

LogicalResult realize(ScanOp operation) {
  OpBuilder builder(operation);
  Location location = operation.getLoc();
  auto states = scratch(builder, location, operation.getInitials());
  auto next = scratch(builder, location, operation.getInitials());
  for (auto [initial, state] : llvm::zip(operation.getInitials(), states)) copy(builder, location, initial, state);
  Value zero = index(builder, location, 0), one = index(builder, location, 1);
  Value extent = builder.create<memref::DimOp>(location, operation.getSources().front(), operation.getAxis());
  auto loop = builder.create<scf::ForOp>(location, zero, extent, one);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value coordinate = loop.getInductionVar();
    if (operation.getReverse())
      coordinate = builder.create<arith::SubIOp>(location, builder.create<arith::SubIOp>(location, extent, one), coordinate);
    int64_t axis = operation.getAxis();
    SmallVector<Value> arguments(states);
    for (Value source : operation.getSources()) arguments.push_back(memberSlice(builder, location, source, {axis}, coordinate));
    llvm::append_range(arguments, operation.getCaptures());
    llvm::append_range(arguments, next);
    if (failed(instantiateHelper(builder, operation.getCombine(), arguments))) return failure();
    for (auto [prefix, output] : llvm::zip(operation.getInclusive() ? next : states, operation.getOutputs()))
      copy(builder, location, prefix, memberSlice(builder, location, output, {axis}, coordinate));
    for (auto [source, destination] : llvm::zip(next, states)) copy(builder, location, source, destination);
  }
  for (Value value : next) builder.create<memref::DeallocOp>(location, value);
  for (Value value : states) builder.create<memref::DeallocOp>(location, value);
  operation.erase();
  return success();
}

} // namespace

LogicalResult realizeSliceCollectives(func::FuncOp function) {
  SmallVector<Operation *> operations;
  function.walk<WalkOrder::PostOrder>([&](Operation *operation) {
    if (isa<SliceReduceOp>(operation)) operations.push_back(operation);
    else if (auto scan = dyn_cast<ScanOp>(operation); scan && scan.isDestinationPassing())
      operations.push_back(operation);
  });
  for (Operation *operation : operations) {
    if (auto reduction = dyn_cast<SliceReduceOp>(operation)) {
      if (failed(realize(reduction))) return failure();
    } else if (failed(realize(cast<ScanOp>(operation)))) return failure();
  }
  return success();
}

} // namespace intent::cpu
