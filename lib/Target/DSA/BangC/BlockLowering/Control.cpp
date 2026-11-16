#include "Lowering.h"

namespace intent::bangc::block_lowering {
SmallVector<Type> Lowering::leaves(Type type) {
  if (auto record = dyn_cast<block::RecordType>(type)) {
    SmallVector<Type> result;
    for (Attribute attribute : record.getFieldTypes()) llvm::append_range(result, leaves(cast<TypeAttr>(attribute).getValue()));
    return result;
  }
  return {type};
}

SmallVector<Value> Lowering::flatten(ValueRange originals) {
  SmallVector<Value> result;
  for (Value original : originals) {
    if (isa<block::RecordType>(original.getType())) llvm::append_range(result, records.lookup(original));
    else result.push_back(get(original));
  }
  return result;
}

FailureOr<SmallVector<Value>> Lowering::slots(ValueRange originals) {
  SmallVector<Value> result;
  for (Value original : originals) {
    for (Type type : leaves(original.getType())) {
      Value slot = allocate(type);
      if (!slot) return failure();
      result.push_back(slot);
    }
  }
  return result;
}

LogicalResult Lowering::publish(ValueRange originals, ValueRange storage) {
  SmallVector<Value> snapshots;
  auto inputs = flatten(originals);
  SmallVector<Type> types;
  for (Value original : originals) llvm::append_range(types, leaves(original.getType()));
  if (inputs.size() != storage.size()) return emitError(loc, "BANG C control product schema changed");
  // Parallel assignment: materialize every next value before modifying any
  // carried slot, including permutations of loop state and tuple fields.
  for (auto [input, slot, type] : llvm::zip_equal(inputs, storage, types)) {
    Value value = input;
    if (!value) return emitError(loc, "BANG C control yielded an unbound value");
    if (llvm::is_contained(storage, value) && value != slot) {
      Value snapshot = allocate(type);
      if (!snapshot || failed(copy(value, snapshot))) return failure();
      value = snapshot;
    }
    snapshots.push_back(value);
  }
  for (auto [value, slot] : llvm::zip_equal(snapshots, storage))
    if (failed(copy(value, slot))) return failure();
  return success();
}

LogicalResult Lowering::bindState(ValueRange originals, ValueRange storage) {
  unsigned position = 0;
  for (Value original : originals) {
    SmallVector<Value> fields;
    for (Type type : leaves(original.getType())) {
      Value slot = storage[position++], value = slot;
      if (!isa<block::FragmentType>(type)) {
        auto scalar = castScalarValue(b, loc, read(slot, {}), type);
        if (failed(scalar)) return failure();
        value = *scalar;
      }
      fields.push_back(value);
    }
    if (isa<block::RecordType>(original.getType())) records[original] = fields;
    else values.map(original, fields.front());
  }
  return success();
}

LogicalResult Lowering::control(Operation *operation) {
  auto storage = slots(operation->getResults());
  if (failed(storage)) return failure();
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    if (failed(publish(loop.getInitArgs(), *storage))) return failure();
    auto lowered = b.create<scf::ForOp>(loc, get(loop.getLowerBound()), get(loop.getUpperBound()), get(loop.getStep()));
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(lowered.getBody());
      values.map(loop.getInductionVar(), lowered.getInductionVar());
      if (failed(bindState(loop.getRegionIterArgs(), *storage)) ||
          failed(operations(*loop.getBody())) ||
          failed(publish(loop.getBody()->getTerminator()->getOperands(), *storage))) return failure();
    }
  } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
    Value predicate = get(branch.getCondition());
    if (isa<MemRefType>(predicate.getType())) predicate = read(predicate, {});
    auto lowered = b.create<scf::IfOp>(loc, predicate, !branch.getElseRegion().empty());
    auto lower = [&](Block *original, Block *target) -> LogicalResult {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(target);
      if (failed(operations(*original))) return failure();
      return publish(original->getTerminator()->getOperands(), *storage);
    };
    if (failed(lower(branch.thenBlock(), lowered.thenBlock())) ||
        (!branch.getElseRegion().empty() && failed(lower(branch.elseBlock(), lowered.elseBlock())))) return failure();
  } else if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
    if (flatten(loop.getInits()).size() != storage->size())
      return loop.emitError("BANG C while requires a consistent carried state schema");
    if (failed(publish(loop.getInits(), *storage))) return failure();
    auto lowered = b.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
    lowered.getBefore().push_back(new Block());
    lowered.getAfter().push_back(new Block());
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&lowered.getBefore().front());
      if (failed(bindState(loop.getBefore().front().getArguments(), *storage)) ||
          failed(operations(loop.getBefore().front()))) return failure();
      auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
      Value predicate = get(condition.getCondition());
      if (isa<MemRefType>(predicate.getType())) predicate = read(predicate, {});
      if (failed(publish(condition.getArgs(), *storage))) return failure();
      b.create<scf::ConditionOp>(loc, predicate, ValueRange{});
      b.setInsertionPointToStart(&lowered.getAfter().front());
      if (failed(bindState(loop.getAfter().front().getArguments(), *storage)) ||
          failed(operations(loop.getAfter().front())) ||
          failed(publish(loop.getAfter().front().getTerminator()->getOperands(), *storage))) return failure();
      b.create<scf::YieldOp>(loc);
    }
  } else return operation->emitError("unhandled Block control operation");
  return bindState(operation->getResults(), *storage);
}
}
