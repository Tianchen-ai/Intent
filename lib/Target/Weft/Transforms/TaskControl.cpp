#include "TaskConversion.h"
#include "ScalarValues.h"
#include "Views.h"
#include "Intent/Analysis/ControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
using namespace task_detail;

namespace {

Type ownerValueType(Type type) {
  if (isa<wk::ValueType>(type)) return type;
  auto empty = DenseI64ArrayAttr::get(type.getContext(), {});
  return wk::ValueType::get(type.getContext(), type, empty, empty);
}

} // namespace

FailureOr<SmallVector<Value>> TaskConversion::writtenEnclosingLocals(Operation *scope) {
  SmallVector<Value> result;
  auto effects = storage->effects(scope);
  if (!effects.complete)
    return scope->emitError("Weft control requires complete storage effects"), failure();
  for (const cpu::StorageEffect &entry : effects.entries) {
    if (!isa<MemoryEffects::Write>(entry.effect.getEffect())) continue;
    Value memory = entry.effect.getValue();
    // Atomic operations separately expose their addressed storage effects.
    // Their resource-less effects order surrounding accesses; they do not
    // mutate every private value carried by the enclosing structured region.
    if (!memory && isa<cpu::AtomicLoadOp, cpu::AtomicStoreOp,
                       cpu::AtomicRMWOp, cpu::AtomicCompareExchangeOp>(entry.operation))
      continue;
    if (!memory)
      return scope->emitError("Weft control write has no storage target"), failure();
    Value owner = localRoot(memory);
    if (owner && isLocal(owner)) {
      Operation *definition = owner.getDefiningOp();
      if (auto argument = dyn_cast<BlockArgument>(owner))
        definition = argument.getOwner()->getParentOp();
      if (definition != scope && !scope->isProperAncestor(definition) &&
          !llvm::is_contained(result, owner)) result.push_back(owner);
      continue;
    }
    auto origins = storage->origins(memory);
    if (!origins.complete)
      return scope->emitError("Weft control write has unresolved storage origins"), failure();
    for (Value root : origins.values)
      if (isLocal(root) && !scope->isProperAncestor(root.getDefiningOp()) &&
          !llvm::is_contained(result, root)) result.push_back(root);
  }
  return result;
}

LogicalResult TaskConversion::checkCarry(Value root, const LocalValue &before, Operation *scope) {
  if (isa<wk::ValueType>(before.value.getType()) && failed(materializeLocal(root)))
    return failure();
  auto found = locals.find(root);
  if (found == locals.end() || found->second.value.getType() != before.value.getType() ||
      found->second.sizes.size() != before.sizes.size() ||
      !llvm::all_of(llvm::zip(found->second.sizes, before.sizes), [&](auto bounds) {
        return sameBound(std::get<0>(bounds), std::get<1>(bounds));
      }))
    return scope->emitError("Weft control carry must preserve its complete initialized region and type");
  return success();
}

LogicalResult TaskConversion::bindLocalReference(Value memory, Operation *scope) {
  Value root = fullLocalOwner(memory);
  if (!root)
    return scope->emitError("Weft task control requires a complete private storage owner; dynamic external descriptors belong to host control");
  if (memory != root) localReferences[memory] = root;
  return success();
}

bool TaskConversion::immutableBorrow(Value memory) {
  auto view = storage->externalView(memory);
  if (!view || view.getAccess() != 0) return false;
  auto effects = storage->effects(currentTask);
  if (!effects.complete || effects.ordered) return false;
  for (const cpu::StorageEffect &entry : effects.entries) {
    Value affected = entry.effect.getValue();
    if (isa<MemoryEffects::Write>(entry.effect.getEffect()) &&
        (!affected || !storage->disjoint(affected, memory))) return false;
    // Native ownership has already proved that a mixed-origin conditional
    // release never frees its borrowed alternative. A direct release of the
    // borrowed storage does not have that ownership distinction.
    if (isa<MemoryEffects::Free>(entry.effect.getEffect()) &&
        (!affected || storage->uniqueOrigin(affected) == memory)) return false;
  }
  return true;
}

bool TaskConversion::completePrivateValue(Value memory) {
  auto origins = storage->origins(memory);
  if (!origins.complete || origins.values.empty()) return false;
  for (Value origin : origins.values) {
    if (immutableBorrow(origin)) continue;
    Operation *allocation = origin.getDefiningOp();
    if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) ||
        !currentTask->isProperAncestor(allocation)) return false;
  }
  llvm::SmallDenseSet<Value> visited;
  std::function<bool(Value)> complete = [&](Value value) {
    if (!visited.insert(value).second) return true;
    auto sources = storage->origins(value);
    // The actual descriptor is admitted at its original projection; a
    // borrowed tile need not cover the public allocation's complete shape.
    if (sources.complete && !sources.values.empty() &&
        llvm::all_of(sources.values, [&](Value origin) { return immutableBorrow(origin); }))
      return true;
    if (!sameStorageShape(memory, value)) return false;
    if (llvm::is_contained(origins.values, value)) return true;
    if (auto cast = value.getDefiningOp<memref::CastOp>())
      return complete(cast.getSource());
    if (auto view = value.getDefiningOp<memref::SubViewOp>()) {
      if (view.getDroppedDims().any()) return false;
      for (auto [axis, size] : llvm::enumerate(view.getMixedSizes()))
        if (!sameBound(view.getMixedOffsets()[axis], b.getIndexAttr(0)) ||
            !sameBound(view.getMixedStrides()[axis], b.getIndexAttr(1)) ||
            !cpu::haveEqualExtents(ValueBoundsConstraintSet::Variable(size),
                ValueBoundsConstraintSet::Variable(view.getSource(), axis))) return false;
      return complete(view.getSource());
    }
    auto incoming = intent::queryControlFlowIncoming(value);
    return incoming.complete && !incoming.edges.empty() &&
        llvm::all_of(incoming.edges, [&](const intent::ControlFlowEdge &edge) {
          return edge.operand && complete(edge.operand->get());
        });
  };
  return complete(memory);
}

LogicalResult TaskConversion::isolateControlValue(Value memory, Operation *scope,
                                 ValueRange boundaries) {
  if (!completePrivateValue(memory))
    return scope->emitError("Weft task control requires complete private values; external runtime descriptors require host control");
  auto origins = storage->origins(memory);
  llvm::SmallDenseSet<Value> incomingAliases, outgoingAliases;
  for (Value boundary : boundaries) {
    if (!isa<MemRefType>(boundary.getType())) continue;
    auto aliases = storage->aliases(boundary);
    if (!aliases.complete)
      return scope->emitError("private control value has an unresolved storage escape");
    auto &allowed = isa<BlockArgument>(boundary) ? incomingAliases : outgoingAliases;
    allowed.insert(aliases.values.begin(), aliases.values.end());
  }
  auto taskEffects = storage->effects(currentTask);
  auto writesAliases = [&](const auto &aliases) {
    return llvm::any_of(taskEffects.entries, [&](const cpu::StorageEffect &entry) {
      return isa<MemoryEffects::Write>(entry.effect.getEffect()) &&
          (!entry.effect.getValue() || aliases.contains(entry.effect.getValue()));
    });
  };
  bool boundaryWrites = writesAliases(incomingAliases) || writesAliases(outgoingAliases);
  for (Value origin : origins.values) {
    auto aliases = storage->aliases(origin);
    if (!aliases.complete)
      return scope->emitError("private control storage has an unresolved alias or escape");
    Operation *definition = origin.getDefiningOp();
    bool createdInside = definition && scope->isProperAncestor(definition);
    bool mutatesIncoming = llvm::any_of(storage->effects(scope).entries,
        [&](const cpu::StorageEffect &entry) {
          if (!isa<MemoryEffects::Write>(entry.effect.getEffect())) return false;
          Value target = entry.effect.getValue();
          if (!target) return true;
          if (incomingAliases.contains(target)) return true;
          return !createdInside && llvm::is_contained(storage->origins(target).values, origin);
        });
    for (Value alias : aliases.values)
      for (Operation *user : alias.getUsers()) {
        if (!currentTask->isProperAncestor(user)) continue;
        if (user == scope || isa<RegionBranchOpInterface,
                RegionBranchTerminatorOpInterface>(user) ||
            cpu::isStorageAliasOperation(user)) continue;
        auto effects = storage->effects(user);
        if (!effects.complete || effects.ordered)
          return scope->emitError("private control storage has an ordered or unknown observer");
        bool observes = llvm::any_of(effects.entries, [&](const cpu::StorageEffect &entry) {
          return (!entry.effect.getValue() || entry.effect.getValue() == alias) &&
              isa<MemoryEffects::Read, MemoryEffects::Write>(entry.effect.getEffect());
        });
        if (!observes) continue;
        if (scope->isProperAncestor(user)) {
          if (!isa<scf::IfOp>(scope) && !createdInside &&
              !incomingAliases.contains(alias) && mutatesIncoming)
            return scope->emitError("private control value has a separately observable incoming alias");
        } else {
          Operation *position = scope->getBlock()->findAncestorOpInBlock(*user);
          if (position && position->isBeforeInBlock(scope)) continue;
          bool writes = llvm::any_of(effects.entries, [&](const cpu::StorageEffect &entry) {
            return (!entry.effect.getValue() || entry.effect.getValue() == alias) &&
                isa<MemoryEffects::Write>(entry.effect.getEffect());
          });
          if (!outgoingAliases.contains(alias) && (writes || boundaryWrites))
            return scope->emitError("private control value retains a separately observable outgoing alias");
        }
      }
  }
  // Distinct live descriptors cannot become independent value states when
  // writes through one may be observed through another descriptor.
  for (Value other : boundaries) {
    if (other == memory || !isa<MemRefType>(other.getType())) continue;
    auto argument = dyn_cast<BlockArgument>(memory);
    auto otherArgument = dyn_cast<BlockArgument>(other);
    bool simultaneous = argument && otherArgument
        ? argument.getOwner() == otherArgument.getOwner()
        : !argument && !otherArgument;
    if (!simultaneous || storage->disjoint(memory, other)) continue;
    llvm::SmallDenseSet<Value> sharedAliases;
    for (Value value : {memory, other}) {
      auto aliases = storage->aliases(value);
      sharedAliases.insert(aliases.values.begin(), aliases.values.end());
    }
    if (writesAliases(sharedAliases))
      return scope->emitError("private control values share an observable mutable storage identity");
  }
  return success();
}

LogicalResult TaskConversion::prepareControlValues(Operation *scope, ValueRange boundaries) {
  for (Value value : boundaries) {
    if (!isa<MemRefType>(value.getType())) continue;
    Value owner = fullLocalOwner(value);
    if (owner && !scope->isProperAncestor(owner.getDefiningOp())) {
      if (failed(bindLocalReference(value, scope))) return failure();
    } else {
      if (failed(isolateControlValue(value, scope, boundaries))) return failure();
      controlOwners.insert(value);
    }
  }
  return success();
}

FailureOr<SmallVector<Type>> TaskConversion::controlTypes(ValueRange boundaries) {
  SmallVector<Type> result;
  for (Value value : boundaries) {
    if (!isa<MemRefType>(value.getType())) result.push_back(nativeScalarType(value.getType()));
    else if (controlOwners.contains(value)) {
      auto type = resultType(value);
      if (failed(type)) return failure();
      result.push_back(ownerValueType(*type));
    }
  }
  return result;
}

Value TaskConversion::controlOwner(Type type, Location loc, Value initial) {
  SmallVector<Value> extents;
  for (int64_t dimension : shape(type))
    extents.push_back(dimension < 0 ? shapeValues[-dimension - 1] : index(loc, dimension));
  Value owner = b.create<wk::EmptyOp>(loc, type, extents);
  if (initial)
    owner = b.create<wk::UpdateOp>(loc, type, owner, initial, ValueRange{},
        b.getArrayAttr(SmallVector<Attribute>(shape(type).size(), b.getStringAttr("all"))));
  return owner;
}

Value TaskConversion::nativeOwner(Value value) {
  while (auto update = value.getDefiningOp<wk::UpdateOp>()) value = update.getInput();
  return value.getDefiningOp<wk::NewOp>() || value.getDefiningOp<wk::EmptyOp>() ||
      value.getDefiningOp<wk::MaterializeOp>()
      ? value : Value{};
}

bool TaskConversion::canReuseControlOwner(Value input, Value boundary, Operation *scope) {
  Value root = fullLocalOwner(input);
  if (!root || !locals.count(root)) return false;
  auto aliases = storage->aliases(root), forwarded = storage->aliases(boundary);
  if (!aliases.complete || !forwarded.complete) return false;
  llvm::SmallDenseSet<Value> transferred(forwarded.values.begin(), forwarded.values.end());
  for (Value alias : aliases.values) {
    if (transferred.contains(alias)) continue;
    for (Operation *user : alias.getUsers()) {
      if (user == scope || cpu::isStorageAliasOperation(user) ||
          isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface>(user)) continue;
      auto effects = storage->effects(user);
      if (!effects.complete || effects.ordered) return false;
      bool observes = llvm::any_of(effects.entries, [&](const cpu::StorageEffect &entry) {
        return (!entry.effect.getValue() || entry.effect.getValue() == alias) &&
            isa<MemoryEffects::Read, MemoryEffects::Write>(entry.effect.getEffect());
      });
      if (!observes) continue;
      Operation *position = scope->getBlock()->findAncestorOpInBlock(*user);
      if (!position || position == scope || !position->isBeforeInBlock(scope)) return false;
    }
  }
  return true;
}

void TaskConversion::ownControlInputs(SmallVectorImpl<Value> &initial, ValueRange inputs,
                      ValueRange boundaries, Operation *scope) {
  llvm::SmallDenseSet<Value> reused;
  unsigned position = 0;
  for (auto [input, boundary] : llvm::zip_equal(inputs, boundaries)) {
    if (isa<MemRefType>(boundary.getType()) && !controlOwners.contains(boundary)) continue;
    Value &value = initial[position++];
    if (!isa<MemRefType>(boundary.getType()) || !isa<wk::ValueType>(value.getType())) continue;
    Value owner = nativeOwner(value);
    if (!owner || reused.contains(owner) || !canReuseControlOwner(input, boundary, scope))
      value = controlOwner(value.getType(), scope->getLoc(), value);
    else reused.insert(owner);
  }
}

FailureOr<Value> TaskConversion::alignControlValue(Value value, Type target, Location loc) {
  if (value.getType() == target) return value;
  if (isa<wk::ValueType>(value.getType()) && isa<wk::ValueType>(target) &&
      element(value.getType()) == element(target) && shape(value.getType()) == shape(target))
    return Value(b.create<wk::ReshapeOp>(loc, target, value, array(axes(value.getType()))));
  return alignValue(value, target, loc);
}

FailureOr<SmallVector<Value>> TaskConversion::controlValues(ValueRange inputs, ValueRange boundaries,
                                           ValueRange owners) {
  SmallVector<Value> result;
  for (auto [input, boundary] : llvm::zip_equal(inputs, boundaries)) {
    if (!isa<MemRefType>(input.getType())) result.push_back(values.lookup(input));
    else if (controlOwners.contains(boundary)) {
      if (Value owner = fullLocalOwner(input); owner && !locals.count(owner))
        if (failed(bindLocalReference(input, boundary.getParentBlock()->getParentOp())))
          return failure();
      auto supplied = cast<MemRefType>(input.getType()).getRank() == 0
          ? readNative(input) : read(input);
      auto type = resultType(boundary);
      if (failed(supplied) || failed(type)) return failure();
      *type = ownerValueType(*type);
      if (!owners.empty() && owners[result.size()]) *type = owners[result.size()].getType();
      auto aligned = alignControlValue(*supplied, *type, input.getLoc());
      if (failed(aligned)) return failure();
      result.push_back(*aligned);
    }
  }
  if (!owners.empty()) {
    auto exactOwner = [&](Value value, Value owner) {
      while (auto update = value.getDefiningOp<wk::UpdateOp>()) value = update.getInput();
      return value == owner;
    };
    auto referencesOwner = [&](Value value) {
      while (true) {
        if (llvm::is_contained(owners, value)) return true;
        if (auto update = value.getDefiningOp<wk::UpdateOp>()) value = update.getInput();
        else if (auto extract = value.getDefiningOp<wk::ExtractOp>()) value = extract.getInput();
        else if (auto reshape = value.getDefiningOp<wk::ReshapeOp>()) value = reshape.getInput();
        else return false;
      }
    };
    // Region exits are parallel assignments. Preserve every incoming owner
    // used by another output before any update changes its physical contents.
    for (auto [value, owner] : llvm::zip_equal(result, owners))
      if (owner && isa<wk::ValueType>(value.getType()) &&
          !exactOwner(value, owner) && referencesOwner(value))
        value = controlOwner(value.getType(), value.getLoc(), value);
    for (auto [value, owner] : llvm::zip_equal(result, owners))
      if (owner && isa<wk::ValueType>(value.getType()) && !exactOwner(value, owner))
        value = b.create<wk::UpdateOp>(value.getLoc(), value.getType(), owner, value, ValueRange{},
            b.getArrayAttr(SmallVector<Attribute>(shape(value.getType()).size(), b.getStringAttr("all"))));
  }
  return result;
}

LogicalResult TaskConversion::mapControlValues(ValueRange source, ValueRange target) {
  unsigned position = 0;
  for (Value value : source) {
    if (!isa<MemRefType>(value.getType())) values.map(value, target[position++]);
    else if (controlOwners.contains(value)) {
      Value supplied = target[position++];
      auto type = resultType(value);
      if (failed(type)) return failure();
      *type = ownerValueType(*type);
      auto aligned = alignControlValue(supplied, *type, value.getLoc());
      if (failed(aligned)) return failure();
      supplied = *aligned;
      SmallVector<OpFoldResult> sizes;
      for (int64_t extent : shape(supplied.getType()))
        sizes.push_back(extent < 0 ? OpFoldResult(shapeValues[-extent - 1])
                                   : OpFoldResult(b.getIndexAttr(extent)));
      locals[value] = {supplied, std::move(sizes)};
    }
  }
  return success();
}

LogicalResult TaskConversion::lower(scf::IfOp conditional) {
  Location loc = conditional.getLoc();
  if (failed(prepareControlValues(conditional, conditional.getResults()))) return failure();
  auto written = writtenEnclosingLocals(conditional);
  if (failed(written)) return failure();
  auto &roots = *written;
  for (Value root : roots)
    if (failed(materializeLocal(root))) return failure();
  auto savedLocals = locals;
  auto nativeTypes = controlTypes(conditional.getResults());
  if (failed(nativeTypes)) return failure();
  auto types = *nativeTypes;
  unsigned explicitResults = types.size();
  SmallVector<Value> owners;
  for (Type type : types)
    owners.push_back(isa<wk::ValueType>(type) ? controlOwner(type, loc) : Value{});
  for (Value root : roots) {
    auto found = savedLocals.find(root);
    if (found == savedLocals.end()) {
      auto type = resultType(root);
      if (failed(type)) return failure();
      types.push_back(*type);
    } else types.push_back(found->second.value.getType());
  }
  auto target = b.create<scf::IfOp>(loc, types, values.lookup(conditional.getCondition()),
      !types.empty() || !conditional.getElseRegion().empty());
  for (bool then : {true, false}) {
    Block *destination = then ? target.thenBlock() : target.getElseRegion().empty() ? nullptr : target.elseBlock();
    if (!destination) continue;
    locals = savedLocals;
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(destination);
    Block *source = then ? conditional.thenBlock() : conditional.getElseRegion().empty() ? nullptr : conditional.elseBlock();
    if (source && failed(block(*source))) return failure();
    auto supplied = source
        ? controlValues(source->getTerminator()->getOperands(), conditional.getResults(), owners)
        : FailureOr<SmallVector<Value>>(SmallVector<Value>{});
    if (failed(supplied)) return failure();
    auto results = *supplied;
    for (Value root : roots) {
      if (!locals.count(root)) return conditional.emitError("conditional local result is not initialized on every branch");
      if (failed(materializeLocal(root))) return failure();
      if (savedLocals.count(root) && failed(checkCarry(root, savedLocals.lookup(root), conditional))) return failure();
      results.push_back(locals.lookup(root).value);
    }
    if (!types.empty()) b.create<scf::YieldOp>(loc, results);
  }
  locals = savedLocals;
  if (failed(mapControlValues(conditional.getResults(), target.getResults().take_front(explicitResults)))) return failure();
  for (auto [root, value] : llvm::zip(roots, target.getResults().drop_front(explicitResults)))
    if (locals.count(root)) locals[root].value = value;
    else if (failed(write(root, value))) return failure();
  return success();
}

LogicalResult TaskConversion::lower(scf::ForOp loop) {
  Location loc = loop.getLoc();
  SmallVector<Value> boundaries(loop.getRegionIterArgs());
  llvm::append_range(boundaries, loop.getResults());
  if (failed(prepareControlValues(loop, boundaries))) return failure();
  auto supplied = controlValues(loop.getInitArgs(), loop.getRegionIterArgs());
  if (failed(supplied)) return failure();
  auto initial = *supplied;
  ownControlInputs(initial, loop.getInitArgs(), loop.getRegionIterArgs(), loop);
  unsigned explicitResults = initial.size();
  SmallVector<Value> roots;
  auto written = writtenEnclosingLocals(loop);
  if (failed(written)) return failure();
  for (Value root : *written) {
    if (!locals.count(root)) continue;
    if (failed(materializeLocal(root))) return failure();
    roots.push_back(root);
    initial.push_back(locals.lookup(root).value);
  }
  auto savedLocals = locals;
  auto target = b.create<scf::ForOp>(loc, values.lookup(loop.getLowerBound()),
      values.lookup(loop.getUpperBound()), values.lookup(loop.getStep()), initial);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(target.getBody());
    values.map(loop.getInductionVar(), target.getInductionVar());
    if (failed(mapControlValues(loop.getRegionIterArgs(), target.getRegionIterArgs().take_front(explicitResults)))) return failure();
    for (auto [root, value] : llvm::zip(roots, target.getRegionIterArgs().drop_front(explicitResults)))
      locals[root].value = value;
    if (failed(block(*loop.getBody()))) return failure();
    auto supplied = controlValues(loop.getBody()->getTerminator()->getOperands(), loop.getRegionIterArgs(),
        target.getRegionIterArgs().take_front(explicitResults));
    if (failed(supplied)) return failure();
    auto results = *supplied;
    for (Value root : roots) {
      if (failed(checkCarry(root, savedLocals.lookup(root), loop))) return failure();
      results.push_back(locals.lookup(root).value);
    }
    if (!initial.empty()) b.create<scf::YieldOp>(loc, results);
  }
  locals = std::move(savedLocals);
  if (failed(mapControlValues(loop.getResults(), target.getResults().take_front(explicitResults)))) return failure();
  for (auto [root, value] : llvm::zip(roots, target.getResults().drop_front(explicitResults)))
    locals[root].value = value;
  return success();
}

LogicalResult TaskConversion::lower(scf::WhileOp loop) {
  Location loc = loop.getLoc();
  SmallVector<Value> boundaries(loop.getBeforeArguments());
  llvm::append_range(boundaries, loop.getAfterArguments());
  llvm::append_range(boundaries, loop.getResults());
  if (failed(prepareControlValues(loop, boundaries))) return failure();
  auto supplied = controlValues(loop.getInits(), loop.getBeforeArguments());
  auto nativeTypes = controlTypes(loop.getResults());
  if (failed(supplied) || failed(nativeTypes)) return failure();
  auto initial = *supplied;
  ownControlInputs(initial, loop.getInits(), loop.getBeforeArguments(), loop);
  unsigned explicitInputs = initial.size();
  auto types = *nativeTypes;
  unsigned explicitResults = types.size();
  if (initial.size() != types.size() ||
      !llvm::all_of(llvm::zip(initial, types), [](auto pair) {
        Type left = std::get<0>(pair).getType(), right = std::get<1>(pair);
        return left == right || (isa<wk::ValueType>(left) && isa<wk::ValueType>(right) &&
            element(left) == element(right) && shape(left) == shape(right));
      }))
    return loop.emitError("Weft while requires one stable physical state schema across both regions");
  for (auto [value, type] : llvm::zip(initial, types)) type = value.getType();
  auto written = writtenEnclosingLocals(loop);
  if (failed(written)) return failure();
  SmallVector<Value> roots;
  for (Value root : *written)
    if (locals.count(root)) {
      if (failed(materializeLocal(root))) return failure();
      roots.push_back(root);
      initial.push_back(locals.lookup(root).value);
      types.push_back(locals.lookup(root).value.getType());
    }
  auto savedLocals = locals;
  auto target = b.create<scf::WhileOp>(loc, types, initial);
  Block &before = target.getBefore().emplaceBlock();
  Block &after = target.getAfter().emplaceBlock();
  for (Value value : initial) before.addArgument(value.getType(), loc);
  for (Type type : types) after.addArgument(type, loc);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&before);
    if (failed(mapControlValues(loop.getBeforeArguments(), before.getArguments().take_front(explicitInputs)))) return failure();
    for (auto [root, value] : llvm::zip(roots, before.getArguments().drop_front(explicitInputs)))
      locals[root].value = value;
    if (failed(block(loop.getBefore().front()))) return failure();
    auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
    auto supplied = controlValues(condition.getArgs(), loop.getAfterArguments(),
        before.getArguments().take_front(explicitInputs));
    if (failed(supplied)) return failure();
    auto results = *supplied;
    for (Value root : roots) {
      if (failed(checkCarry(root, savedLocals.lookup(root), loop))) return failure();
      results.push_back(locals.lookup(root).value);
    }
    b.create<scf::ConditionOp>(loc, values.lookup(condition.getCondition()), results);
  }
  locals = savedLocals;
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&after);
    if (failed(mapControlValues(loop.getAfterArguments(), after.getArguments().take_front(explicitResults)))) return failure();
    for (auto [root, value] : llvm::zip(roots, after.getArguments().drop_front(explicitResults)))
      locals[root].value = value;
    if (failed(block(loop.getAfter().front()))) return failure();
    auto supplied = controlValues(loop.getAfter().front().getTerminator()->getOperands(), loop.getBeforeArguments(),
        after.getArguments().take_front(explicitResults));
    if (failed(supplied)) return failure();
    auto results = *supplied;
    for (Value root : roots) {
      if (failed(checkCarry(root, savedLocals.lookup(root), loop))) return failure();
      results.push_back(locals.lookup(root).value);
    }
    b.create<scf::YieldOp>(loc, results);
  }
  locals = std::move(savedLocals);
  if (failed(mapControlValues(loop.getResults(), target.getResults().take_front(explicitResults)))) return failure();
  for (auto [root, value] : llvm::zip(roots, target.getResults().drop_front(explicitResults)))
    locals[root].value = value;
  return success();
}

} // namespace intent::weft_provider
