#include "ContractionDetail.h"
#include "../Value/ScopePlacement.h"
#include "Intent/Dialect/GPU/Analysis/MemoryEffects.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include <functional>

using namespace mlir;

namespace intent::gpu::contraction {
namespace {

// A mutually exclusive path remains exclusive between row iterations only
// when the condition does not depend on this traversal.
bool mutuallyExclusiveRows(Operation *first, Operation *second,
                           scf::ForOp loop) {
  llvm::DenseSet<Value> active;
  std::function<bool(Value)> invariant = [&](Value value) {
    Operation *owner = value.getParentBlock()->getParentOp();
    if (owner != loop && !loop->isAncestor(owner)) return true;
    if (!active.insert(value).second) return false;
    Operation *producer = value.getDefiningOp();
    bool result = producer && producer->getNumRegions() == 0 &&
                  isMemoryEffectFree(producer) &&
                  llvm::all_of(producer->getOperands(), invariant);
    active.erase(value);
    return result;
  };
  for (Operation *current = first; current != loop;
       current = current->getParentOp()) {
    auto branch = dyn_cast<scf::IfOp>(current->getParentOp());
    if (!branch || !invariant(branch.getCondition())) continue;
    for (Operation *other = second; other != loop;
         other = other->getParentOp())
      if (other->getParentOp() == branch &&
          other->getBlock() != current->getBlock()) return true;
  }
  return false;
}

// Reconstruct only the write address, predicate and conditional path. Numeric
// epilogues run in the first phase, where every original input is still intact.
class WritebackSlice {
public:
  WritebackSlice(scf::ForOp loop) : loop(loop), dominance(loop->getParentOp()) {}

  bool accepts(Value value) {
    if (!value || value == loop.getInductionVar()) return true;
    if (invariants.contains(value)) return true;
    if (dominance.dominates(value, loop)) {
      invariants.insert(value);
      return true;
    }
    if (accepted.contains(value)) return true;
    auto operation = value.getDefiningOp();
    if (!operation || !loop->isProperAncestor(operation) ||
        operation->getNumRegions() || !isMemoryEffectFree(operation))
      return false;
    if (!llvm::all_of(operation->getOperands(),
                      [&](Value operand) { return accepts(operand); }))
      return false;
    accepted.insert(value);
    return true;
  }

  FailureOr<Value> materialize(OpBuilder &builder, Value value,
                               IRMapping &mapping) {
    if (!value) return value;
    if (Value replacement = mapping.lookupOrNull(value)) return replacement;
    // Preflight recorded these unchanged SSA captures before moving the loop.
    // Do not reuse dominance facts after adding the two conditional phases.
    if (invariants.contains(value)) return value;
    Operation *operation = value.getDefiningOp();
    if (!operation || !loop->isProperAncestor(operation) ||
        operation->getNumRegions() || !isMemoryEffectFree(operation))
      return failure();
    for (Value operand : operation->getOperands()) {
      auto replacement = materialize(builder, operand, mapping);
      if (failed(replacement)) return failure();
      mapping.map(operand, *replacement);
    }
    builder.clone(*operation, mapping);
    return mapping.lookupOrNull(value);
  }

private:
  scf::ForOp loop;
  DominanceInfo dominance;
  llvm::DenseSet<Value> invariants, accepted;
};

bool dependsOn(Value value, Value coordinate) {
  if (value == coordinate) return true;
  Operation *producer = value.getDefiningOp();
  return producer && producer->getNumRegions() == 0 &&
      llvm::any_of(producer->getOperands(),
                   [&](Value operand) { return dependsOn(operand, coordinate); });
}

FailureOr<Value> transportCapture(OpBuilder &builder, Value value,
                                  Value unboundCoordinate, IRMapping &mapping) {
  if (Value mapped = mapping.lookupOrNull(value)) return mapped;
  if (value == unboundCoordinate) return failure();
  Operation *producer = value.getDefiningOp();
  if (!producer) return value;
  SmallVector<Value> operands;
  bool changed = false;
  for (Value operand : producer->getOperands()) {
    auto replacement = transportCapture(builder, operand, unboundCoordinate, mapping);
    if (failed(replacement)) return failure();
    operands.push_back(*replacement);
    changed |= operand != *replacement;
  }
  if (!changed) return value;
  if (producer->getNumRegions() || !isMemoryEffectFree(producer)) return failure();
  for (auto [source, selected] : llvm::zip(producer->getOperands(), operands))
    mapping.map(source, selected);
  builder.clone(*producer, mapping);
  return mapping.lookupOrNull(value);
}

Value overlapCondition(OpBuilder &builder, func::FuncOp kernel,
                       Value first, Value second, Location location) {
  if (first == second)
    return builder.create<arith::ConstantIntOp>(location, 1, 1);
  for (ViewOverlapOp overlap : kernel.getOps<ViewOverlapOp>())
    if ((overlap.getLhs() == first && overlap.getRhs() == second) ||
        (overlap.getLhs() == second && overlap.getRhs() == first))
      return overlap.getResult();
  OpBuilder entry(&kernel.front(), kernel.front().begin());
  return entry.create<ViewOverlapOp>(location, entry.getI1Type(), first, second);
}

} // namespace

LogicalResult preserveRuntimeContractionSnapshot(
    scf::ForOp loop, MakeRangeOp rows, MakeRangeOp columns,
    Value selectedColumns, Value columnValidity, Value rowWidth,
    Value columnWidth, Value rowWorker, Value columnWorker,
    OpBuilder::Listener *listener) {
  auto kernel = loop->getParentOfType<func::FuncOp>();
  SmallVector<Operation *> reads;
  SmallVector<StoreOp> stores;
  WalkResult effects = loop.walk([&](Operation *operation) {
    auto access = dyn_cast<AccessOpInterface>(operation);
    if (!access) return WalkResult::advance();
    if (auto store = dyn_cast<StoreOp>(operation)) {
      stores.push_back(store);
      return WalkResult::advance();
    }
    auto effects = getEffectsRecursively(operation);
    if (!effects || !hasOnlyReadEffects(operation))
      return WalkResult::interrupt();
    if (llvm::any_of(*effects, [](const MemoryEffects::EffectInstance &effect) {
          return isa<MemoryEffects::Read>(effect.getEffect());
        })) reads.push_back(operation);
    return WalkResult::advance();
  });
  if (effects.wasInterrupted())
    return loop.emitOpError("runtime row replay contains an unordered or unknown access");

  ResourceAliasAnalysis aliases;
  llvm::SetVector<std::pair<Value, Value>> overlaps;
  for (Operation *read : reads)
    for (StoreOp store : stores) {
      if (mutuallyExclusiveRows(read, store, loop) ||
          placement::independentMemoryEffects(read, store, aliases)) continue;
      auto memory = cast<AccessOpInterface>(read).getAccessResourceOperand().get();
      auto input = dyn_cast<BlockArgument>(memory);
      auto output = dyn_cast<BlockArgument>(store.getResource());
      if (!input || !output || input.getOwner() != &kernel.front() ||
          output.getOwner() != &kernel.front() || !getPublicView(input) ||
          !getPublicView(output))
        return store.emitOpError(
            "runtime row replay has no external overlap predicate for its snapshot");
      overlaps.insert({memory, store.getResource()});
    }
  if (overlaps.empty()) return success();

  auto group = loop->getParentOfType<ExecutionGroupOp>();
  if (!group)
    return loop.emitOpError("runtime row snapshot has no current execution group");
  SmallVector<Value> groupCoordinates;
  SmallVector<Attribute> groupShape;
  for (auto [axis, coordinate] : llvm::enumerate(group.getCoordinates())) {
    if (coordinate == rowWorker || coordinate == columnWorker) continue;
    groupCoordinates.push_back(coordinate);
    groupShape.push_back(group.getLaunchExtents()[axis]);
  }

  auto selected = selectedColumns.getDefiningOp<MakeRangeOp>();
  if (!selected || !dependsOn(selected.getStart(), columnWorker))
    return loop.emitOpError(
        "runtime row snapshot has no exact column-worker coordinate origin");

  WritebackSlice writeback(loop);
  SmallVector<SmallVector<Block *>> paths;
  for (StoreOp store : stores) {
    auto path = placement::conditionalPath(loop.getBody(), store);
    auto type = dyn_cast<ViewType>(store.getResource().getType());
    if (failed(path) || !type || !writeback.accepts(store.getResource()) ||
        !writeback.accepts(store.getValid()) ||
        !llvm::all_of(store.getCoordinates(),
                      [&](Value value) { return writeback.accepts(value); }) ||
        llvm::any_of(*path, [&](Block *block) {
          return !writeback.accepts(
              cast<scf::IfOp>(block->getParentOp()).getCondition());
        }))
      return store.emitOpError(
          "runtime row snapshot writeback requires stable coordinates and predicates");
    paths.push_back(std::move(*path));
  }

  OpBuilder builder(loop, listener);
  Location location = loop.getLoc();
  Value overlapsOutput;
  for (auto [input, output] : overlaps) {
    Value condition = overlapCondition(builder, kernel, input, output, location);
    overlapsOutput = overlapsOutput
        ? Value(builder.create<BinaryOp>(location, builder.getI1Type(),
              overlapsOutput, condition, BinaryOperator::LogicalOr))
        : condition;
  }
  SmallVector<Value> snapshots;
  for (StoreOp store : stores) {
    auto output = cast<ViewType>(store.getResource().getType());
    auto value = cast<FragmentType>(store.getValue().getType());
    SmallVector<Attribute> shape(groupShape);
    llvm::append_range(shape, output.getLayout().getExtents());
    snapshots.push_back(createInvocationBuffer(
        kernel, location, value.getElementType(), builder.getArrayAttr(shape),
        value.getOwner()).getResult());
  }
  auto choice = builder.create<scf::IfOp>(location, overlapsOutput, true);
  // The original bounded traversal belongs to the disjoint branch unchanged.
  loop->moveBefore(choice.elseBlock()->getTerminator());
  builder.setInsertionPoint(choice.thenBlock()->getTerminator());
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value groupValidity;
  for (auto [coordinate, shape] : llvm::zip(groupCoordinates, groupShape)) {
    Value extent = builder.create<PhysicalExprOp>(
        location, builder.getIndexType(), cast<PhysicalExprAttr>(shape));
    Value lower = builder.create<CompareOp>(location, builder.getI1Type(),
        coordinate, zero, ComparePredicate::Ge);
    Value upper = builder.create<CompareOp>(location, builder.getI1Type(),
        coordinate, extent, ComparePredicate::Lt);
    Value bounded = builder.create<BinaryOp>(location, builder.getI1Type(),
        lower, upper, BinaryOperator::LogicalAnd);
    groupValidity = groupValidity
        ? Value(builder.create<BinaryOp>(location, builder.getI1Type(),
              groupValidity, bounded, BinaryOperator::LogicalAnd))
        : bounded;
  }
  Value rowOwner = builder.create<CompareOp>(location, builder.getI1Type(),
      rowWorker, zero, ComparePredicate::Eq);
  Value columnOwner = builder.create<CompareOp>(location, builder.getI1Type(),
      columnWorker, zero, ComparePredicate::Eq);
  Value owner = builder.create<BinaryOp>(location, builder.getI1Type(),
      rowOwner, columnOwner, BinaryOperator::LogicalAnd);
  auto owned = builder.create<scf::IfOp>(location, owner, false);
  builder.setInsertionPoint(owned.thenBlock()->getTerminator());

  auto emitPhase = [&](bool publish) -> LogicalResult {
    auto columnLoop = builder.create<scf::ForOp>(location,
        columns.getLogicalStart(), columns.getLogicalStop(), columnWidth);
    OpBuilder columnBuilder(columnLoop.getBody()->getTerminator(), listener);
    auto newColumns = columnBuilder.create<MakeRangeOp>(
        location, selected.getResult().getType(), columnLoop.getInductionVar(),
        selected.getExtent(), selected.getStep(), selected.getLogicalStart(),
        selected.getLogicalStop(), selected.getSourceId(), selected.getSourceAxis(),
        selected.getDerived());
    inheritRangeAuthority(newColumns, selected);
    auto valid = rangeBoundsValidity(columnBuilder, location,
        selected.getResult().getType(), cast<FragmentType>(columnValidity.getType()),
        newColumns, selected.getLogicalStop());
    IRMapping mapping;
    mapping.map(selectedColumns, newColumns.getResult());
    mapping.map(columnValidity, valid);
    mapping.map(selected.getStart(), columnLoop.getInductionVar());
    mapping.map(rowWorker, zero);
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(loop.getRegion(), captures);
    for (Value capture : captures) {
      auto replacement = transportCapture(columnBuilder, capture, columnWorker, mapping);
      if (failed(replacement))
        return loop.emitOpError(
            "runtime row snapshot cannot transport an external column capture");
      mapping.map(capture, *replacement);
    }
    if (!publish) {
      auto computation = cast<scf::ForOp>(columnBuilder.clone(*loop, mapping));
      computation.setLowerBound(rows.getLogicalStart());
      computation.setStep(rowWidth);
      SmallVector<StoreOp> writes;
      computation.walk([&](StoreOp store) { writes.push_back(store); });
      if (writes.size() != snapshots.size())
        return computation.emitOpError("contraction snapshot lost its output occurrences");
      for (auto [store, snapshot] : llvm::zip(writes, snapshots)) {
        if (groupValidity) {
          OpBuilder atStore(store, listener);
          auto validity = materializeValidityConjunction(atStore, store.getLoc(),
              store.getValid(), groupValidity,
              cast<FragmentType>(store.getValue().getType()));
          if (failed(validity))
            return store.emitOpError("contraction snapshot group bounds cannot be projected");
          store.getValidMutable().assign(*validity);
        }
        store.getResourceMutable().assign(snapshot);
        SmallVector<Value> coordinates(groupCoordinates);
        llvm::append_range(coordinates, store.getCoordinates());
        SmallVector<int64_t> axes;
        for (unsigned axis = 0; axis < groupCoordinates.size(); ++axis)
          axes.push_back(axis);
        for (int64_t axis : store.getSourceAxes())
          axes.push_back(axis + groupCoordinates.size());
        store.getCoordinatesMutable().assign(coordinates);
        cast<AccessOpInterface>(store.getOperation()).setAccessSourceAxes(axes);
        // This is a compiler-private snapshot write. The original observable
        // effect remains on the external publication in the second phase.
        store->removeAttr(originAttr);
      }
      return success();
    }
    auto rowLoop = columnBuilder.create<scf::ForOp>(location,
        rows.getLogicalStart(), rows.getLogicalStop(), rowWidth);
    mapping.map(loop.getInductionVar(), rowLoop.getInductionVar());
    OpBuilder rowBuilder(rowLoop.getBody()->getTerminator(), listener);
    placement::ConditionalPlacement conditions;
    for (auto [index, store] : llvm::enumerate(stores)) {
      auto condition = [&](OpBuilder &nested, Operation *original) -> FailureOr<Value> {
        IRMapping local(mapping);
        return writeback.materialize(nested,
            cast<scf::IfOp>(original).getCondition(), local);
      };
      auto body = [&](OpBuilder &nested) -> LogicalResult {
        IRMapping local(mapping);
        SmallVector<Value> coordinates;
        for (Value coordinate : store.getCoordinates()) {
          auto selected = writeback.materialize(nested, coordinate, local);
          if (failed(selected)) return failure();
          coordinates.push_back(*selected);
        }
        auto valid = writeback.materialize(nested, store.getValid(), local);
        auto resource = writeback.materialize(nested, store.getResource(), local);
        auto type = cast<FragmentType>(store.getValue().getType());
        auto fill = materializeZeroFragment(nested, location, type);
        if (failed(valid) || failed(resource) || failed(fill)) return failure();
        SmallVector<Value> sourceCoordinates(groupCoordinates);
        llvm::append_range(sourceCoordinates, coordinates);
        SmallVector<int64_t> sourceAxes;
        for (unsigned axis = 0; axis < groupCoordinates.size(); ++axis)
          sourceAxes.push_back(axis);
        for (int64_t axis : store.getSourceAxes())
          sourceAxes.push_back(axis + groupCoordinates.size());
        Value snapshotValidity = *valid;
        if (groupValidity) {
          auto projected = materializeValidityConjunction(
              nested, location, *valid, groupValidity, type);
          if (failed(projected)) return failure();
          snapshotValidity = *projected;
        }
        Value value = nested.create<LoadOp>(location, type, snapshots[index],
            sourceCoordinates, snapshotValidity, *fill, sourceAxes);
        auto output = nested.create<StoreOp>(location, *resource, coordinates,
            value, *valid, store.getSourceAxes());
        if (Attribute origin = store->getAttr(originAttr))
          output->setAttr(originAttr, origin);
        return success();
      };
      if (failed(conditions.emit(rowBuilder, paths[index], condition, body)))
        return store.emitOpError("contraction snapshot writeback could not be materialized");
    }
    return success();
  };
  if (failed(emitPhase(false)) || failed(emitPhase(true))) return failure();
  return success();
}

} // namespace intent::gpu::contraction
