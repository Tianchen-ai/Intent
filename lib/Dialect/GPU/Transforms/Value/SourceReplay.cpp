#include "SourceReplay.h"
#include "ReplayInsertion.h"
#include "ReplayPolicy.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include <algorithm>

using namespace mlir;

namespace intent::gpu {
namespace {

// Equal physical shapes make this comparison independent of parameter bindings.
bool noLargerSnapshots(Value source, ArrayRef<Value> snapshots) {
  auto type = dyn_cast<FragmentType>(source.getType());
  if (!type) return false;
  auto words = [](Type element) -> uint64_t {
    return std::max(1u, ((element.isIndex() ? 64u :
        element.getIntOrFloatBitWidth()) + 31) / 32);
  };
  uint64_t remaining = words(type.getElementType());
  for (Value value : snapshots) {
    auto saved = dyn_cast<FragmentType>(value.getType());
    if (!saved || saved.getShape() != type.getShape()) return false;
    uint64_t current = words(saved.getElementType());
    if (current > remaining) return false;
    remaining -= current;
  }
  return true;
}

FailureOr<SmallVector<Value>> sourceSnapshots(
    Value value, PhysicalSourceAxis source, int64_t dimension,
    std::optional<unsigned> sourceAxis, Operation *anchor,
    PhysicalReplayScope scope, const IRMapping &bindings,
    const ReplayPolicy *sharedReuse = nullptr) {
  auto kernel = anchor->getParentOfType<func::FuncOp>();
  if (!kernel) return failure();
  PhysicalProgramAnalysis analysis(kernel);
  DominanceInfo dominance(kernel);
  IRMapping preserved(bindings);
  SmallVector<Value> snapshots;
  auto wholeSource = [&]() -> FailureOr<SmallVector<Value>> {
    if (!sourceAxis || !isa<FragmentType>(value.getType()) ||
        !dominance.dominates(value, anchor)) return failure();
    auto range = queryExactLogicalRange(analysis.axisRanges(value, *sourceAxis));
    if (failed(range) || !isUnitStepRange(*range)) return failure();
    return SmallVector<Value>{value};
  };
  while (true) {
    auto proof = analysis.replayAt(value, source, scope, true, anchor,
                                   preserved, dimension);
    if (proof.isReplayable()) break;
    bool changed = false;
    for (Operation *blocker : proof.blockers) {
      auto load = dyn_cast<LoadOp>(blocker);
      if (!load || preserved.lookupOrNull(load.getResult()) ||
          canReplayReadAt(load, anchor) ||
          !dominance.dominates(load.getResult(), anchor)) continue;
      auto axis = queryFragmentAxis(load.getType(), source, dimension);
      if (!axis.isExact() || (load.getFill() && !uniformScalarSource(load.getFill())))
        continue;
      auto range = queryExactLogicalRange(
          analysis.axisRanges(load.getResult(), axis.fragmentAxis));
      if (failed(range) || !isUnitStepRange(*range)) continue;
      preserved.map(load.getResult(), load.getResult());
      snapshots.push_back(load.getResult());
      changed = true;
    }
    if (!changed) return wholeSource();
  }
  if (!snapshots.empty()) {
    ReplayPolicy reuse(kernel, ValueRange{value}, {anchor}, [&](Value current) {
      return queryFragmentAxis(current.getType(), source, dimension).isExact();
    });
    const ReplayPolicy &policy = sharedReuse ? *sharedReuse : reuse;
    // Replacing a still-live root would retain both that root and its inputs.
    if (!policy.removesProducer(value) || !noLargerSnapshots(value, snapshots) ||
        policy.duplicatesExpensiveWork(value, &preserved)) return wholeSource();
  }
  return snapshots;
}

// Preserve a computed prefix that already survives this rewrite. Binding its
// current SSA value must close the entire remaining suffix proof; provenance
// of its reads alone never grants permission to read those buffers again.
LogicalResult bindComputedSnapshot(
    OpBuilder &builder, Value root, PhysicalSourceAxis source, int64_t dimension,
    PhysicalExprAttr extent, Value coordinates, Operation *anchor,
    PhysicalReplayScope scope, AxisMapAttr resultMapping, IRMapping &mapping,
    const ReplayPolicy &reuse) {
  auto kernel = anchor->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  auto original = analysis.replayAt(root, source, scope, true, anchor,
                                    mapping, dimension);
  if (original.isReplayable() || !reuse.removesProducer(root) ||
      !llvm::any_of(original.blockers, [&](Operation *blocker) {
        auto load = dyn_cast<LoadOp>(blocker);
        return load && !canReplayReadAt(load, anchor);
      })) return success();

  DominanceInfo dominance(kernel);
  SmallVector<Value> pending{root};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second || mapping.lookupOrNull(value)) continue;
    Operation *producer = value.getDefiningOp();
    if (!producer || producer->getNumRegions() ||
        !isPhysicalReplayNode(producer, PhysicalReplayScope::ValueGraph,
                              /*allowAccesses=*/false)) continue;
    auto axis = queryFragmentAxis(value.getType(), source, dimension);
    if (value != root && axis.isExact() && !reuse.removesProducer(value) &&
        noLargerSnapshots(root, ArrayRef<Value>{value}) &&
        dominance.dominates(value, anchor)) {
      IRMapping preserved(mapping);
      preserved.map(value, value);
      if (analysis.replayAt(root, source, scope, true, anchor, preserved,
                            dimension).isReplayable() &&
          !reuse.duplicatesExpensiveWork(root, &preserved)) {
        auto saved = materializeSnapshotSlice(builder, value, axis.fragmentAxis,
            extent, coordinates, anchor, resultMapping);
        if (failed(saved)) return failure();
        if (*saved) {
          mapping.map(value, *saved);
          return success();
        }
      }
    }
    llvm::append_range(pending, producer->getOperands());
  }
  return success();
}

FailureOr<Value> sliceSourceSnapshot(
    OpBuilder &builder, Location location, Value value, unsigned axis,
    PhysicalExprAttr extent, Value coordinate, Operation *anchor,
    AxisMapAttr resultMapping, bool preserveReadFill) {
  auto result = materializeRetainedSlice(builder, location, value, axis, extent,
                                        coordinate, anchor, resultMapping);
  if (failed(result)) return failure();
  auto gather = (*result).getDefiningOp<GatherOp>();
  OpBuilder fillBuilder(gather);
  Value fill;
  if (auto load = value.getDefiningOp<LoadOp>();
      preserveReadFill && load && load.getFill())
    fill = uniformScalarSource(load.getFill());
  if (fill) {
    DominanceInfo dominance(anchor->getParentOfType<func::FuncOp>());
    if (!availableAtInsertionPoint(fill, fillBuilder, dominance)) return failure();
    Value padded = fillBuilder.create<SplatOp>(location, gather.getType(), fill);
    gather.getFillMutable().assign(padded);
  }
  return result;
}

} // namespace

LogicalResult bindSourceReplay(
    OpBuilder &builder, Location location, Value root, PhysicalSourceAxis source,
    int64_t dimension, std::optional<unsigned> sourceAxis,
    PhysicalExprAttr extent, Value coordinate, Operation *anchor,
    PhysicalReplayScope scope, AxisMapAttr resultMapping,
    IRMapping &resultBindings, const ReplayPolicy *sharedReuse) {
  auto kernel = anchor->getParentOfType<func::FuncOp>();
  ReplayPolicy localReuse(kernel, ValueRange{root}, {anchor}, [&](Value current) {
    return queryFragmentAxis(current.getType(), source, dimension).isExact();
  });
  const ReplayPolicy &reuse = sharedReuse ? *sharedReuse : localReuse;
  ReplayInsertion insertion(builder);
  IRMapping mapping(resultBindings);
  if (failed(bindComputedSnapshot(builder, root, source, dimension, extent,
      coordinate, anchor, scope, resultMapping, mapping, reuse))) return failure();
  auto snapshots = sourceSnapshots(root, source, dimension, sourceAxis, anchor,
                                   scope, mapping, &reuse);
  if (failed(snapshots)) return failure();
  for (Value value : *snapshots) {
    auto selected = queryFragmentAxis(value.getType(), source, dimension);
    std::optional<unsigned> axis = value == root ? sourceAxis :
        selected.isExact() ? std::optional<unsigned>(selected.fragmentAxis) : std::nullopt;
    if (!axis) return failure();
    auto saved = sliceSourceSnapshot(builder, location, value, *axis, extent,
        coordinate, anchor, resultMapping, value != root);
    if (failed(saved)) return failure();
    mapping.map(value, *saved);
  }
  resultBindings = std::move(mapping);
  insertion.commit();
  return success();
}

LogicalResult prepareSourceReplay(Value value, unsigned axis, Operation *anchor,
                                 PhysicalReplayScope scope) {
  auto type = dyn_cast<FragmentType>(value.getType());
  if (!type || axis >= type.getShape().size() || !anchor) return failure();
  auto kernel = anchor->getParentOfType<func::FuncOp>();
  if (!kernel) return failure();
  // Native operands already realized need no coverage preparation. Actual
  // slicing retains its own coordinate proof at the consumer emission point.
  if (PhysicalProgramAnalysis(kernel).axisRealization(value, axis).physicalized)
    return success();
  auto relation = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
  auto snapshots = sourceSnapshots(value, sourceAxisIdentity(relation),
      relation.getDimensionId(), axis, anchor, scope, IRMapping{});
  if (failed(snapshots))
    return anchor->emitError("source supply cannot establish its saved snapshot boundary")
        << "; source=" << value << "; axis=" << axis;
  for (Value saved : *snapshots) {
    auto projection = queryFragmentAxis(saved.getType(), sourceAxisIdentity(relation),
                                        relation.getDimensionId());
    if (saved != value && !projection.isExact()) return failure();
    unsigned selected = saved == value ? axis : projection.fragmentAxis;
    if (!PhysicalProgramAnalysis(kernel).axisRealization(saved, selected).physicalized &&
        failed(realizeFullCoverageDimension(kernel, saved, selected))) return failure();
  }
  return success();
}

FailureOr<Value> materializeSourceValue(
    OpBuilder &builder, Location location, Value value, PhysicalSourceAxis source,
    PhysicalExprAttr extent, IRMapping &resultMapping, Operation *anchor,
    ReplayMaterializationOptions options, Attribute sourceTail,
    const ReplayPolicy &reuse) {
  auto type = dyn_cast<FragmentType>(value.getType());
  auto selected = queryFragmentAxis(value.getType(), source);
  std::optional<unsigned> axis = options.fragmentAxis ? options.fragmentAxis :
      selected.isExact() ? std::optional<unsigned>(selected.fragmentAxis) : std::nullopt;
  if (!type || !axis || *axis >= type.getShape().size()) return failure();
  int64_t dimension = cast<AxisMapAttr>(type.getAxisMaps()[*axis]).getDimensionId();
  if (options.traversalRanges.empty()) return failure();
  MakeRangeOp range = options.traversalRanges.front();
  Value coordinates = resultMapping.lookupOrNull(range.getResult());
  if (!coordinates) return failure();
  IRMapping mapping(resultMapping);
  ReplayInsertion insertion(builder);
  if (failed(bindSourceReplay(builder, location, value, source, dimension, axis,
      extent, coordinates, anchor, options.scope, options.segmentMapping,
      mapping, &reuse))) return failure();
  if (Value saved = mapping.lookupOrNull(value);
      saved && !resultMapping.lookupOrNull(value) && sourceTail &&
      sourceTail != uniformZero(type.getElementType())) {
    auto savedType = cast<FragmentType>(saved.getType());
    auto fill = materializeScalarConstant(builder, location, sourceTail, type.getElementType());
    auto tail = projectPredicateToFragmentAxis(builder, location, options.segmentTail,
                                               savedType, *axis);
    if (failed(fill) || failed(tail)) return failure();
    Value padded = builder.create<SplatOp>(location, savedType, *fill);
    Value selected = builder.create<SelectOp>(location, savedType, *tail, saved, padded);
    mapping.map(value, selected);
  }
  if (failed(reuse.bindSlices(builder, value, source, dimension, extent, coordinates,
      anchor, mapping, options.segmentMapping))) return failure();
  auto result = mapping.lookupOrNull(value)
      ? FailureOr<Value>(mapping.lookup(value))
      : materializeReplayedValue(builder, location, value, source, extent, mapping,
                                 anchor, options);
  if (failed(result)) return failure();
  resultMapping = std::move(mapping);
  insertion.commit();
  return result;
}

FailureOr<Value> materializeSourceRanges(
    OpBuilder &builder, Location location, Value value, PhysicalExprAttr extent,
    ArrayRef<MakeRangeOp> roots, Value coordinates, IRMapping &resultMapping,
    Operation *anchor) {
  if (roots.empty() || !anchor) return failure();
  auto kernel = anchor->getParentOfType<func::FuncOp>();
  auto source = sourceAxisIdentity(roots.front());
  auto dimension = queryRangeDimension(roots.front());
  if (failed(dimension) || !llvm::all_of(roots, [&](MakeRangeOp root) {
        auto current = queryRangeDimension(root);
        return sourceAxisIdentity(root) == source && succeeded(current) &&
               *current == *dimension;
      })) return failure();
  auto projection = PhysicalProgramAnalysis(kernel).rangeAxes(value, roots);
  std::optional<unsigned> axis = projection.isExact() && projection.fragmentAxes.size() == 1
      ? std::optional<unsigned>(projection.fragmentAxes.front()) : std::nullopt;
  IRMapping mapping(resultMapping);
  ReplayInsertion insertion(builder);
  if (failed(bindSourceReplay(builder, location, value, source, *dimension, axis,
      extent, coordinates, anchor, PhysicalReplayScope::ValueGraph, {}, mapping)))
    return anchor->emitError("range source supply cannot bind its saved snapshots")
        << "; source=" << value;
  if (llvm::any_of(value.getUsers(), [&](Operation *user) {
        return user == anchor || anchor->isAncestor(user);
      })) {
    ReplayPolicy reuse(kernel, ValueRange{value}, {anchor}, [&](Value current) {
      return queryFragmentAxis(current.getType(), source, *dimension).isExact();
    });
    if (failed(reuse.bindSlices(builder, value, source, *dimension, extent,
        coordinates, anchor, mapping))) return failure();
  }
  auto result = mapping.lookupOrNull(value)
      ? FailureOr<Value>(mapping.lookup(value))
      : materializeReplayedRanges(builder, location, value, extent, roots,
                                  coordinates, mapping, anchor);
  if (failed(result))
    return anchor->emitError("range source supply could not rebuild its pure suffix")
        << "; source=" << value;
  resultMapping = std::move(mapping);
  insertion.commit();
  return result;
}

FailureOr<Value> materializeSourceRanges(
    OpBuilder &builder, Location location, func::FuncOp kernel, Value value,
    PhysicalSourceAxis source, PhysicalExprAttr extent, MakeRangeOp root,
    Value coordinates, IRMapping &mapping, Operation *anchor) {
  if (!kernel || !(sourceAxisIdentity(root) == source)) return failure();
  return materializeSourceRanges(builder, location, value, extent,
      ArrayRef<MakeRangeOp>{root}, coordinates, mapping, anchor);
}

FailureOr<Value> materializeSourceRead(
    OpBuilder &builder, Location location, LoadOp load, unsigned axis,
    FragmentType resultType, ValueRange coordinates, Value valid, Value fill,
    Value traversalCoordinate, Operation *anchor) {
  if (canReplayReadAt(load, anchor))
    return Value(builder.create<LoadOp>(location, resultType, load.getResource(),
                                       coordinates, valid, fill, load.getSourceAxes()));
  if (!traversalCoordinate) return failure();
  auto extent = cast<PhysicalExprAttr>(resultType.getShape()[axis]);
  return sliceSourceSnapshot(builder, location, load.getResult(), axis, extent,
                              traversalCoordinate, anchor, {}, false);
}

} // namespace intent::gpu
