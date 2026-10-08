#include "../Construction.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <functional>

using namespace mlir;

namespace intent::kir_to_cpu {

LogicalResult Construction::atomicAccess(Operation *operation) {
  auto access = cast<IndexedAccessOpInterface>(operation);
  auto fact = analysis.indexRelation(operation);
  if (failed(fact)) return failure();
  Location loc = operation->getLoc();
  Value target = values.lookup(fact->source);
  Type element = cast<MemRefType>(target.getType()).getElementType();
  auto ordering = operation->getAttrOfType<AtomicOrderingAttr>("ordering");
  SmallVector<Type> resultTypes;
  appendProductLeafTypes(operation->getResultTypes(), resultTypes);
  Type accessType = resultTypes.empty()
      ? access.getStoredValue().getType()
      : resultTypes.front();
  auto tensor = dyn_cast<RankedTensorType>(accessType);
  SmallVector<Value> sizes, members, outputs;
  bool used = llvm::any_of(operation->getResults(), [](Value value) { return !value.use_empty(); });
  if (tensor) {
    Value shapeSource = operation->getNumResults()
        ? operation->getResult(0) : access.getStoredValue();
    SmallVector<unsigned, 2> path;
    bool first = true;
    walkProductLeaves(shapeSource.getType(), [&](Type, ArrayRef<unsigned> fieldPath) {
      if (!first) return;
      path.assign(fieldPath.begin(), fieldPath.end());
      first = false;
    });
    auto extentsOr = extents(shapeSource, loc, path);
    if (failed(extentsOr)) return failure();
    sizes = *extentsOr;
    if (used)
      for (Type result : resultTypes)
        outputs.push_back(emptyTensor(cast<RankedTensorType>(result), sizes, loc));
  }
  auto operand = [&](Value value) {
    return elementAt(values.lookup(value), members, builder, loc);
  };
  std::function<FailureOr<SmallVector<Value>>(unsigned, ValueRange)> traverse =
      [&](unsigned axis, ValueRange carried) -> FailureOr<SmallVector<Value>> {
    if (axis < sizes.size()) {
      auto loop = builder.create<scf::ForOp>(loc, constant(loc, 0), sizes[axis], constant(loc, 1), carried);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      members.push_back(loop.getInductionVar());
      auto results = traverse(axis + 1, loop.getRegionIterArgs());
      members.pop_back();
      if (failed(results)) return failure();
      if (!loop.getBody()->empty() &&
          loop.getBody()->back().hasTrait<OpTrait::IsTerminator>())
        loop.getBody()->back().erase();
      builder.setInsertionPointToEnd(loop.getBody());
      builder.create<scf::YieldOp>(loc, *results);
      return SmallVector<Value>(loop.getResults().begin(), loop.getResults().end());
    }
    auto coordinates = indexedCoordinates(*fact, members, builder, loc);
    if (failed(coordinates)) return failure();
    SmallVector<Value> results;
    if (isa<AtomicLoadOp>(operation)) {
      results.push_back(builder.create<cpu::AtomicLoadOp>(loc, element, target, *coordinates, ordering));
    } else if (isa<AtomicStoreOp>(operation)) {
      builder.create<cpu::AtomicStoreOp>(loc, target, operand(access.getStoredValue()), *coordinates, ordering);
    } else if (isa<AtomicRMWOp>(operation)) {
      results.push_back(builder.create<cpu::AtomicRMWOp>(loc, element, target, operand(access.getStoredValue()),
          *coordinates, ordering, operation->getAttrOfType<AtomicRMWKindAttr>("kind"),
          builder.getBoolAttr(isa<IntegerType>(element) && cast<IntegerType>(element).isUnsigned())));
    } else {
      auto exchange = builder.create<cpu::AtomicCompareExchangeOp>(loc, element, builder.getI1Type(),
          target, operand(access.getCompareValue()), operand(access.getReplacementValue()), *coordinates, ordering);
      llvm::append_range(results, exchange.getResults());
    }
    if (!used) return SmallVector<Value>{};
    if (!tensor) {
      bindValues(operation->getResults(), results);
      return SmallVector<Value>{};
    }
    SmallVector<Value> updated;
    for (auto [value, output] : llvm::zip(results, carried))
      updated.push_back(builder.create<tensor::InsertOp>(loc, value, output, members));
    return updated;
  };
  auto results = traverse(0, outputs);
  if (failed(results)) return failure();
  if (used && tensor) bindValues(operation->getResults(), *results);
  return success();
}

LogicalResult Construction::scatterReduce(ScatterReduceOp operation) {
  auto fact = analysis.indexRelation(operation);
  if (failed(fact)) return failure();
  Location loc = operation.getLoc();
  Value target = values.lookup(fact->source);
  Value input = values.lookup(operation.getValue());
  SmallVector<Value> sizes, members;
  if (auto memory = dyn_cast<RankedTensorType>(input.getType()))
    for (int64_t axis = 0; axis < memory.getRank(); ++axis)
      sizes.push_back(dimension(builder, loc, input, axis));
  std::function<LogicalResult(unsigned)> traverse = [&](unsigned axis) -> LogicalResult {
    if (axis < sizes.size()) {
      auto loop = builder.create<scf::ForOp>(loc, constant(loc, 0), sizes[axis], constant(loc, 1));
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      members.push_back(loop.getInductionVar());
      auto status = traverse(axis + 1);
      members.pop_back();
      return status;
    }
    Value value = elementAt(input, members, builder, loc);
    auto coordinates = indexedCoordinates(*fact, members, builder, loc);
    if (failed(coordinates)) return failure();
    auto update = builder.create<memref::GenericAtomicRMWOp>(loc, target, *coordinates);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&update.getRegion().front());
    Block &combine = operation.getCombine().front();
    values.map(combine.getArgument(0), update.getCurrentValue());
    values.map(combine.getArgument(1), value);
    for (Operation &instruction : combine.without_terminator())
      if (failed(lowerOperation(&instruction))) return failure();
    builder.create<memref::AtomicYieldOp>(loc, values.lookup(combine.getTerminator()->getOperand(0)));
    return success();
  };
  return traverse(0);
}


} // namespace intent::kir_to_cpu
