#include "ContractionDetail.h"
#include "../Value/ReplayPolicy.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>

using namespace mlir;

namespace intent::gpu::contraction {

std::optional<ContractionEpilogue> collectContractionEpilogue(Value root) {
  ContractionEpilogue result;
  result.root = root;
  llvm::SmallPtrSet<Operation *, 16> nodes, stores;
  SmallVector<Value> pending{root};
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    // Inspect every operand occurrence, including repeated uses by one node.
    // A store may consume the contraction only as its stored value.
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto store = dyn_cast<StoreOp>(user)) {
        if (&use != &store.getValueMutable())
          return std::nullopt;
        stores.insert(user);
        continue;
      }
      if (!isa<CastOp, BitcastOp, BroadcastOp, UnaryOp, BinaryOp, CompareOp,
               SelectOp>(user) ||
          (!isa<CastOp, BroadcastOp>(user) &&
           user->getBlock() != root.getParentBlock()))
        return std::nullopt;
      if (auto broadcast = dyn_cast<BroadcastOp>(user)) {
        auto source = dyn_cast<FragmentType>(value.getType());
        auto target = dyn_cast<FragmentType>(broadcast.getResult().getType());
        if (!source || !target || source.getShape() != target.getShape())
          return std::nullopt;
        auto projection = queryAxisProjection(source, target);
        if (!projection.isExact() ||
            llvm::any_of(llvm::enumerate(projection.targetToSource),
                         [](auto entry) {
                           return !entry.value() ||
                                  *entry.value() != entry.index();
                         }))
          return std::nullopt;
      }
      if (nodes.insert(user).second)
        pending.push_back(user->getResult(0));
    }
  }
  if (stores.empty())
    return std::nullopt;

  auto kernel = root.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return std::nullopt;
  // Preserve the original store order, including stores in nested scopes.
  kernel.walk([&](StoreOp store) {
    if (stores.contains(store))
      result.stores.push_back(store);
  });
  llvm::SmallPtrSet<Operation *, 16> ordered;
  std::function<void(Operation *)> append = [&](Operation *operation) {
    if (!nodes.contains(operation) || !ordered.insert(operation).second)
      return;
    for (Value operand : operation->getOperands())
      if (Operation *producer = operand.getDefiningOp())
        append(producer);
    result.operations.push_back(operation);
  };
  for (StoreOp store : result.stores)
    if (Operation *producer = store.getValue().getDefiningOp())
      append(producer);
  // Dead side branches are part of the replaced closure as well.
  for (Operation *operation : nodes)
    append(operation);
  return result;
}

FailureOr<IRMapping> scalarCaptureBindings(Value value, Operation *anchor) {
  auto kernel = anchor->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  DominanceInfo dominance(kernel);
  IRMapping bindings;
  llvm::DenseSet<Value> visited;
  std::function<bool(Value)> collect = [&](Value current) {
    if (dominance.properlyDominates(current, anchor)) {
      bindings.map(current, current);
      return true;
    }
    if (!visited.insert(current).second)
      return true;
    if (isa<FragmentType, RecordType>(current.getType()))
      return false;
    Operation *producer = current.getDefiningOp();
    return producer && producer->getNumRegions() == 0 &&
           llvm::all_of(producer->getOperands(), collect);
  };
  return collect(value) ? FailureOr<IRMapping>(std::move(bindings))
                        : FailureOr<IRMapping>(failure());
}

namespace {

bool available(Value value, OpBuilder &builder, DominanceInfo &dominance) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return dominance.dominates(argument.getOwner(), builder.getInsertionBlock());
  Operation *producer = value.getDefiningOp();
  return producer && dominance.properlyDominates(
                         producer->getBlock(), producer->getIterator(),
                         builder.getInsertionBlock(), builder.getInsertionPoint(),
                         /*enclosingOpOk=*/false);
}

FragmentType tileType(FragmentType source, FragmentType tile) {
  return FragmentType::get(source.getContext(), source.getElementType(),
                           tile.getShape(), tile.getAxisMaps(),
                           source.getValidity(), source.getOwner());
}

SmallVector<Value> replayRoots(const ContractionEpilogue &epilogue,
                              Value initialAccumulator) {
  SmallVector<Value> roots;
  for (StoreOp store : epilogue.stores)
    roots.push_back(store.getValue());
  if (initialAccumulator)
    roots.push_back(initialAccumulator);
  return roots;
}

SmallVector<Operation *> replacedConsumers(const ContractionEpilogue &epilogue) {
  SmallVector<Operation *> consumers(epilogue.operations);
  consumers.push_back(epilogue.root.getDefiningOp());
  for (StoreOp store : epilogue.stores)
    consumers.push_back(store);
  return consumers;
}

FailureOr<Value> materializeCapture(
    OpBuilder &builder, func::FuncOp kernel, Value value, FragmentType tile,
    Value rows, Value columns, Operation *insertionAnchor,
    const ReplayPolicy &reuse) {
  Location location = insertionAnchor->getLoc();
  DominanceInfo dominance(kernel);
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment) {
    if (available(value, builder, dominance))
      return value;
    auto bindings = scalarCaptureBindings(value, insertionAnchor);
    auto range = rows.getDefiningOp<MakeRangeOp>();
    if (failed(bindings) || !range)
      return failure();
    return materializeReplayedValue(
        builder, location, value, sourceAxisIdentity(range),
        cast<PhysicalExprAttr>(tile.getShape()[0]), *bindings, insertionAnchor);
  }
  auto target = tileType(fragment, tile);
  if (value.getType() == target && available(value, builder, dominance))
    return value;
  if (auto scalar = scalarSource(value); succeeded(scalar)) {
    if (!available(*scalar, builder, dominance))
      return failure();
    return projectPhysicalValueToSchema(builder, location, *scalar, target);
  }
  if (fragment.getShape().size() != 2)
    return failure();
  Value captured = value;
  SmallVector<Value, 2> coordinates{rows, columns};
  llvm::DenseSet<Value> snapshots;
  for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
    auto replacement = coordinate.getDefiningOp<MakeRangeOp>();
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact ranges = analysis.axisRanges(captured, axis);
    if (ranges.isExact() && ranges.roots.empty() && ranges.blockers.empty())
      continue;
    auto root = queryExactLogicalRange(ranges);
    if (!replacement || failed(root) || !isUnitStepRange(*root) ||
        !samePhysicalScalarExpression(root->getLogicalStart(),
                                      replacement.getLogicalStart()) ||
        !samePhysicalScalarExpression(root->getLogicalStop(),
                                      replacement.getLogicalStop())) {
      insertionAnchor->emitError("result capture has no matching result-axis range")
          << "; axis=" << axis << "; value=" << value
          << "; root=" << (succeeded(root) ? root->getResult() : Value())
          << "; replacement=" << coordinate;
      return failure();
    }
    auto axisMap = cast<AxisMapAttr>(
        cast<FragmentType>(value.getType()).getAxisMaps()[axis]);
    auto source = sourceAxisIdentity(axisMap);
    auto indexType = root->getResult().getType();
    auto extent = cast<PhysicalExprAttr>(tile.getShape()[axis]);
    indexType = FragmentType::get(
        kernel.getContext(), indexType.getElementType(),
        builder.getArrayAttr({extent}), indexType.getAxisMaps(),
        indexType.getValidity(), indexType.getOwner());
    Value range = builder.create<MakeRangeOp>(
        location, indexType, replacement.getStart(), replacement.getExtent(),
        root->getStep(), root->getLogicalStart(), root->getLogicalStop(),
        root->getSourceId(), root->getSourceAxis(), root->getDerived());
    inheritRangeAuthority(range, *root);
    auto predicateType = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), indexType.getShape(),
        indexType.getAxisMaps(), indexType.getValidity(), indexType.getOwner());
    Value tail = rangeBoundsValidity(builder, location, indexType, predicateType,
                                    range, root->getLogicalStop());

    // Each axis selects a different schema. Only retained snapshot sources,
    // never partially replayed values, cross this boundary as identity leaves.
    IRMapping mapping;
    for (Value snapshot : snapshots)
      mapping.map(snapshot, snapshot);
    for (MakeRangeOp sourceRange : ranges.roots)
      mapping.map(sourceRange.getResult(), range);
    llvm::DenseSet<Value> existing;
    for (auto entry : mapping.getValueMap())
      existing.insert(entry.first);
    if (failed(reuse.bindSlices(builder, value, source,
                                axisMap.getDimensionId(), extent, range,
                                insertionAnchor, mapping)))
      return failure();
    for (auto entry : mapping.getValueMap()) {
      if (existing.contains(entry.first))
        continue;
      snapshots.insert(entry.first);
      if (auto gather = entry.second.getDefiningOp<GatherOp>())
        snapshots.insert(gather.getSource());
    }
    ReplayMaterializationOptions options;
    options.fragmentAxis = axis;
    options.segmentTail = tail;
    options.materializeZeroFill = true;
    auto replayed = materializeReplayedValue(
        builder, location, value, source, extent, mapping, insertionAnchor,
        options);
    if (failed(replayed)) {
      insertionAnchor->emitError("result capture axis could not be materialized")
          << "; axis=" << axis << "; value=" << value;
      return failure();
    }
    value = *replayed;
  }
  return projectPhysicalValueToSchema(builder, location, value, target);
}

} // namespace

struct ContractionEpilogueMaterialization::Impl {
  Impl(func::FuncOp kernel, const ContractionEpilogue &epilogue,
       FragmentType tile, Value rows, Value columns, Value initialAccumulator)
      : kernel(kernel), epilogue(epilogue), tile(tile), rows(rows),
        columns(columns),
        reuse(kernel, replayRoots(epilogue, initialAccumulator),
              replacedConsumers(epilogue)),
        nodes(epilogue.operations.begin(), epilogue.operations.end()) {}

  IRMapping &mapping(OpBuilder &builder) {
    IRMapping &result = scopes[builder.getInsertionBlock()].values;
    DominanceInfo dominance(kernel);
    SmallVector<Value> unavailable;
    for (auto entry : result.getValueMap())
      if (!available(entry.second, builder, dominance))
        unavailable.push_back(entry.first);
    for (Value value : unavailable)
      result.erase(value);
    return result;
  }

  FailureOr<Value> storeValue(OpBuilder &builder, StoreOp store, Value blocked,
                              Operation *anchor) {
    DominanceInfo dominance(kernel);
    if (blocked.getType() != tile || !available(blocked, builder, dominance))
      return failure();
    IRMapping &values = mapping(builder);
    Scope &scope = scopes[builder.getInsertionBlock()];
    if (scope.blocked && scope.blocked != blocked)
      for (Operation *operation : epilogue.operations)
        for (Value result : operation->getResults())
          values.erase(result);
    scope.blocked = blocked;
    values.map(epilogue.root, blocked);
    return materialize(builder, store.getValue(), anchor, values);
  }

  FailureOr<Value> capture(OpBuilder &builder, Value value,
                           Operation *anchor, IRMapping &mapping) {
    if (Value previous = mapping.lookupOrNull(value))
      return previous;
    auto result = materializeCapture(builder, kernel, value, tile, rows, columns,
                                     anchor, reuse);
    if (succeeded(result))
      mapping.map(value, *result);
    return result;
  }

  FailureOr<Value> materialize(OpBuilder &builder, Value value,
                               Operation *anchor, IRMapping &mapping) {
    if (Value previous = mapping.lookupOrNull(value))
      return previous;
    Operation *operation = value.getDefiningOp();
    if (!operation || !nodes.contains(operation))
      return capture(builder, value, anchor, mapping);
    for (Value operand : operation->getOperands())
      if (failed(materialize(builder, operand, anchor, mapping)))
        return failure();
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment)
      return failure();
    auto cloned = cloneWithSchema(builder, operation, mapping,
                                   TypeRange{tileType(fragment, tile)});
    if (failed(cloned))
      return failure();
    return cloned->front();
  }

  func::FuncOp kernel;
  const ContractionEpilogue &epilogue;
  FragmentType tile;
  Value rows, columns;
  ReplayPolicy reuse;
  llvm::SmallPtrSet<Operation *, 16> nodes;
  struct Scope {
    IRMapping values;
    Value blocked;
  };
  DenseMap<Block *, Scope> scopes;
};

ContractionEpilogueMaterialization::ContractionEpilogueMaterialization(
    func::FuncOp kernel, const ContractionEpilogue &epilogue, FragmentType tile,
    Value rows, Value columns, Value initialAccumulator)
    : impl(std::make_unique<Impl>(kernel, epilogue, tile, rows, columns,
                                  initialAccumulator)) {}

ContractionEpilogueMaterialization::~ContractionEpilogueMaterialization() = default;

FailureOr<Value> ContractionEpilogueMaterialization::capture(
    OpBuilder &builder, Value value, Operation *anchor) {
  return impl->capture(builder, value, anchor, impl->mapping(builder));
}

FailureOr<Value> ContractionEpilogueMaterialization::storeValue(
    OpBuilder &builder, StoreOp store, Value blocked, Operation *anchor) {
  return impl->storeValue(builder, store, blocked, anchor);
}

void eraseContractionEpilogue(const ContractionEpilogue &epilogue) {
  for (StoreOp store : epilogue.stores)
    store.erase();
  // Keep products themselves alive until their contraction worklist entry is
  // consumed; every obsolete epilogue node is unique and topologically ordered.
  for (Operation *operation : llvm::reverse(epilogue.operations))
    if (operation->use_empty())
      operation->erase();
}

} // namespace intent::gpu::contraction
