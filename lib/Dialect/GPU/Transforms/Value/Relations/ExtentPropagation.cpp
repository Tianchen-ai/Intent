#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include <functional>

using namespace mlir;

namespace intent::gpu {
namespace {

struct AxisOccurrence {
  SmallVector<unsigned> fields;
  unsigned axis;
  bool operator==(const AxisOccurrence &other) const {
    return fields == other.fields && axis == other.axis;
  }
};

FragmentType fragmentAt(Type type, ArrayRef<unsigned> fields) {
  for (unsigned field : fields) {
    auto record = dyn_cast<RecordType>(type);
    if (!record || field >= record.getFieldTypes().size())
      return {};
    type = cast<TypeAttr>(record.getFieldTypes()[field]).getValue();
  }
  return dyn_cast<FragmentType>(type);
}

Type replaceExtent(Type type, const AxisOccurrence &occurrence,
                   PhysicalExprAttr extent, unsigned depth = 0) {
  if (depth != occurrence.fields.size()) {
    auto record = cast<RecordType>(type);
    SmallVector<Attribute> fields(record.getFieldTypes().getValue());
    unsigned field = occurrence.fields[depth];
    fields[field] = TypeAttr::get(replaceExtent(
        cast<TypeAttr>(fields[field]).getValue(), occurrence, extent, depth + 1));
    return RecordType::get(record.getContext(), record.getFieldNames(),
                           ArrayAttr::get(record.getContext(), fields),
                           record.getOwner());
  }
  auto fragment = cast<FragmentType>(type);
  SmallVector<Attribute> shape(fragment.getShape().getValue());
  shape[occurrence.axis] = extent;
  return FragmentType::get(fragment.getContext(), fragment.getElementType(),
                           ArrayAttr::get(fragment.getContext(), shape),
                           fragment.getAxisMaps(), fragment.getValidity(),
                           fragment.getOwner());
}

struct ExtentChange {
  Value value;
  AxisOccurrence occurrence;
  PhysicalExprAttr extent;
  bool inspectProducer = true;
};

// This is one explicit extent decision, not a second relation-closure pass.
// Operation relations are captured before the first mutation of their operands
// or results. All mutations still notify the enclosing RelationWorklist.
class ExtentPropagation {
public:
  ExtentPropagation(Value root, ValueTypeChangeCallback changed,
                    OpBuilder::Listener *listener)
      : root(root), changed(changed), listener(listener) {}

  LogicalResult run(ArrayRef<AxisOccurrence> selected, PhysicalExprAttr extent) {
    for (const AxisOccurrence &axis : selected)
      append(root, axis, extent);
    return drain();
  }

  LogicalResult run(ArrayRef<ExtentChange> selected) {
    for (const ExtentChange &change : selected)
      append(change.value, change.occurrence, change.extent);
    return drain();
  }

private:
  LogicalResult drain() {
    while (!pending.empty()) {
      ExtentChange change = pending.pop_back_val();
      auto fragment = fragmentAt(change.value.getType(), change.occurrence.fields);
      if (!fragment || change.occurrence.axis >= fragment.getShape().size())
        return diagnostic(change.value, "extent decision lost its fragment occurrence");
      auto &history = visited[change.value];
      auto found = llvm::find_if(history, [&](const Visit &visit) {
        return visit.occurrence == change.occurrence &&
               visit.extent == change.extent && visit.type == change.value.getType() &&
               visit.inspectProducer == change.inspectProducer;
      });
      if (found != history.end())
        continue;
      history.push_back({change.occurrence, change.extent, change.value.getType(),
                         change.inspectProducer});
      if (change.value != root && change.value.getDefiningOp<MakeRangeOp>())
        continue;
      if (isSegmentSlice(change.value, change.occurrence.axis))
        continue;
      if (change.occurrence.fields.empty() &&
          isIntroducedReshapeUnitAxis(change.value, change.occurrence.axis))
        continue;

      Operation *definition = change.value.getDefiningOp();
      if (definition && failed(capture(definition)))
        return failure();
      for (OpOperand &use : change.value.getUses())
        if (failed(capture(use.getOwner())))
          return failure();
      Type previous = change.value.getType();
      setPhysicalValueType(change.value,
          replaceExtent(previous, change.occurrence, change.extent), changed);
      if (auto range = change.value.getDefiningOp<MakeRangeOp>();
          range && previous != change.value.getType()) {
        OpBuilder builder(range);
        builder.setListener(listener);
        Value size = change.extent.getKind() == PhysicalExprKind::Constant
            ? Value(builder.create<arith::ConstantIndexOp>(range.getLoc(), change.extent.getValue()))
            : Value(builder.create<PhysicalExprOp>(range.getLoc(), builder.getIndexType(), change.extent));
        range->setOperand(1, size);
      }
      if (failed(propagateProducer(change)))
        return failure();
      for (OpOperand &use : llvm::make_early_inc_range(change.value.getUses()))
        if (failed(propagateUse(change, use, previous)))
          return failure();
    }
    for (Value boundary : boundaries)
      if (failed(projectSchemaBoundary(boundary, changed, listener)))
        return failure();
    return success();
  }

private:
  struct Visit {
    AxisOccurrence occurrence;
    PhysicalExprAttr extent;
    Type type;
    bool inspectProducer;
  };
  struct Snapshot {
    SmallVector<SmallVector<FragmentOperandRelation>, 1> results;
    SmallVector<BroadcastProjection> coordinates;
    SmallVector<Type> operands;
    SmallVector<Type> resultTypes;
  };

  LogicalResult diagnostic(Value value, StringRef message) {
    Operation *owner = value.getDefiningOp();
    if (!owner)
      owner = cast<BlockArgument>(value).getOwner()->getParentOp();
    return owner->emitOpError(message);
  }

  void append(Value value, const AxisOccurrence &occurrence,
              PhysicalExprAttr extent, bool inspectProducer = true) {
    if (!value)
      return;
    auto fragment = fragmentAt(value.getType(), occurrence.fields);
    if (!fragment || occurrence.axis >= fragment.getShape().size())
      return;
    pending.push_back({value, occurrence, extent, inspectProducer});
  }

  void append(Value value, const ExtentChange &change) {
    append(value, change.occurrence, change.extent);
  }

  bool isSegmentSlice(Value value, unsigned axis) {
    auto argument = dyn_cast<BlockArgument>(value);
    auto structured = argument
        ? dyn_cast_or_null<StructuredOpInterface>(argument.getOwner()->getParentOp())
        : StructuredOpInterface();
    if (!structured)
      return false;
    for (const auto &relation : structured.getRegionArgumentRelations(
             *argument.getOwner()->getParent()))
      if (relation.to == argument &&
          relation.kind == StructuredRelationKind::SourceSlice &&
          llvm::is_contained(relation.axes, static_cast<int64_t>(axis)))
        return true;
    return false;
  }

  LogicalResult capture(Operation *operation) {
    if (snapshots.count(operation))
      return success();
    Snapshot snapshot;
    snapshot.operands.assign(operation->getOperandTypes().begin(), operation->getOperandTypes().end());
    snapshot.resultTypes.assign(operation->getResultTypes().begin(), operation->getResultTypes().end());
    if (isa<FragmentOpInterface>(operation)) {
      for (OpResult result : operation->getResults()) {
        auto relations = queryFragmentOperandRelations(result);
        if (failed(relations))
          return operation->emitOpError("extent propagation requires a complete fragment relation");
        snapshot.results.push_back(std::move(*relations));
      }
    }
    if (auto access = dyn_cast<AccessOpInterface>(operation))
      for (unsigned slot = 0; slot < access.getAccessCoordinates().size(); ++slot)
        snapshot.coordinates.push_back(queryAccessCoordinateAxes(access, slot));
    snapshots.try_emplace(operation, std::move(snapshot));
    return success();
  }

  LogicalResult propagateFragmentUse(const ExtentChange &change, OpOperand &use) {
    if (!change.occurrence.fields.empty())
      return success();
    Operation *owner = use.getOwner();
    const auto &snapshot = snapshots.find(owner)->second;
    for (auto [resultNumber, relations] : llvm::enumerate(snapshot.results)) {
      for (const FragmentOperandRelation &relation : relations) {
        if (relation.operandNumber != use.getOperandNumber())
          continue;
        for (const FragmentAxisGroup &group : relation.groups) {
          if (group.kind == FragmentAxisRelationKind::Broadcast ||
              !llvm::is_contained(group.sourceAxes, change.occurrence.axis))
            continue;
          FragmentOperandRelation selected = relation;
          selected.groups.assign(1, group);
          auto transported = transportFragmentResultType(
              ArrayRef<FragmentOperandRelation>{selected}, owner->getOperandTypes(),
              owner->getResult(resultNumber).getType());
          if (failed(transported))
            return owner->emitOpError("selected extent cannot cross its fragment relation");
          auto target = dyn_cast<FragmentType>(*transported);
          if (!target)
            continue;
          for (unsigned axis : group.resultAxes)
            append(owner->getResult(resultNumber), {{}, axis},
                   cast<PhysicalExprAttr>(target.getShape()[axis]),
                   /*inspectProducer=*/false);
        }
      }
    }
    return success();
  }

  LogicalResult propagateFragmentProducer(const ExtentChange &change,
                                         OpResult result) {
    if (!change.occurrence.fields.empty())
      return success();
    Operation *owner = result.getOwner();
    const auto &relations = snapshots.find(owner)->second.results[result.getResultNumber()];
    for (const auto &relation : relations) {
      // Multi-input numerical operations do not authorize rewriting another
      // producer's independent traversal. Shape relations are explicit edges.
      if (!relation.preservesSourceSchema && relations.size() != 1)
        continue;
      const auto *group = relation.groupForResultAxis(change.occurrence.axis);
      if (!group || group->kind == FragmentAxisRelationKind::Broadcast)
        continue;
      FragmentOperandRelation selected = relation;
      selected.groups.assign(1, *group);
      Value operand = owner->getOperand(relation.operandNumber);
      auto transported = transportFragmentOperandType(selected, result.getType(), operand.getType());
      if (failed(transported))
        return owner->emitOpError("selected extent cannot invert its fragment relation");
      auto source = dyn_cast<FragmentType>(*transported);
      if (!source)
        continue;
      for (unsigned axis : group->sourceAxes)
        append(operand, {{}, axis}, cast<PhysicalExprAttr>(source.getShape()[axis]));
    }
    return success();
  }

  void appendGroup(const StructuredSchemaGroup &group,
                   const ExtentChange &change) {
    append(group.producer->get(), change);
    for (BlockArgument value : group.arguments) append(value, change);
    for (Value value : group.results) append(value, change);
    for (OpOperand *yield : group.yields) append(yield->get(), change);
  }

  void appendCollectiveComponent(StructuredOpInterface structured,
                                 unsigned component,
                                 const ExtentChange &change) {
    if (component >= structured.getIdentities().size())
      return;
    append(structured->getResult(component), change);
    append(structured.getCombineLhs()[component], change);
    append(structured.getCombineRhs()[component], change);
    append(structured.getCombineYields()[component], change);
  }

  LogicalResult propagateEmission(RegionScanOp scan, unsigned output,
                                  Value target, const ExtentChange &change) {
    auto axis = queryRegionScanEmissionAxis(scan, output, change.occurrence.fields);
    if (failed(axis))
      return scan.emitOpError("emission extent has no unique member-axis relation");
    if (*axis != change.occurrence.axis)
      append(target, change);
    return success();
  }

  LogicalResult propagateStructuredTarget(const ExtentChange &change,
                                          StructuredOpInterface structured) {
    bool ordinary = structured.getStructuredKind() == StructuredOpKind::Reduce ||
                    structured.getStructuredKind() == StructuredOpKind::Scan;
    if (ordinary) {
      if (auto result = dyn_cast<OpResult>(change.value)) {
        appendCollectiveComponent(structured, result.getResultNumber(), change);
        if (structured.getStructuredKind() == StructuredOpKind::Reduce) {
          Value source = structured.getSources()[result.getResultNumber()];
          auto type = fragmentAt(source.getType(), change.occurrence.fields);
          auto reduced = structured.getIterationAxes();
          unsigned freeAxis = 0;
          for (unsigned axis = 0; type && axis < type.getShape().size(); ++axis) {
            if (llvm::is_contained(reduced, static_cast<int64_t>(axis)))
              continue;
            if (freeAxis++ == change.occurrence.axis) {
              AxisOccurrence occurrence = change.occurrence;
              occurrence.axis = axis;
              append(source, occurrence, change.extent);
              break;
            }
          }
        }
      } else if (auto argument = dyn_cast<BlockArgument>(change.value)) {
        unsigned count = structured.getIdentities().size();
        if (argument.getArgNumber() < 2 * count)
          appendCollectiveComponent(structured, argument.getArgNumber() % count, change);
      }
    } else {
      for (const auto &group : queryStructuredSchemaGroups(structured))
        if (llvm::is_contained(group.arguments, change.value) ||
            llvm::is_contained(group.results, change.value))
          appendGroup(group, change);
    }
    if (auto argument = dyn_cast<BlockArgument>(change.value)) {
      for (const auto &relation : structured.getRegionArgumentRelations(
               *argument.getOwner()->getParent())) {
        if (relation.to != argument)
          continue;
        if (relation.kind == StructuredRelationKind::Capture ||
            (relation.kind == StructuredRelationKind::SourceSlice &&
             !llvm::is_contained(relation.axes, static_cast<int64_t>(change.occurrence.axis))))
          append(relation.from, change);
      }
    }
    if (auto scan = dyn_cast<RegionScanOp>(structured.getOperation()))
      if (auto result = dyn_cast<OpResult>(change.value);
          result && result.getResultNumber() < scan.getEmittedResults().size())
        return propagateEmission(scan, result.getResultNumber(),
            structured.getEmitYields()[result.getResultNumber()], change);
    return success();
  }

  LogicalResult propagateProducer(const ExtentChange &change) {
    Value value = change.value;
    Operation *owner = value.getDefiningOp();
    if (auto extract = dyn_cast_or_null<ExtractOp>(owner)) {
      AxisOccurrence occurrence = change.occurrence;
      occurrence.fields.insert(occurrence.fields.begin(), extract.getField());
      append(extract.getRecord(), occurrence, change.extent);
      boundaries.insert(value);
    } else if (isa_and_nonnull<MakeRecordOp>(owner)) {
      boundaries.insert(value);
    } else if (auto result = dyn_cast<OpResult>(value);
               result && isa<FragmentOpInterface>(owner)) {
      if (change.inspectProducer && failed(propagateFragmentProducer(change, result)))
        return failure();
    }
    if (!owner)
      owner = cast<BlockArgument>(value).getOwner()->getParentOp();
    if (auto structured = dyn_cast<StructuredOpInterface>(owner))
      if (failed(propagateStructuredTarget(change, structured)))
        return failure();
    bool control = isa<RegionBranchOpInterface>(owner);
    if (auto argument = dyn_cast<BlockArgument>(value))
      control |= !argument.getOwner()->isEntryBlock();
    if (!control)
      return success();
    auto incoming = queryControlFlowIncoming(value);
    if (!incoming.complete)
      return owner->emitOpError("cannot retarget a produced or unknown control schema slot");
    boundaries.insert(value);
    for (const auto &edge : incoming.edges) {
      if (!edge.operand)
        return failure();
      auto outgoing = queryControlFlowOutgoing(*edge.operand);
      if (!outgoing.complete)
        return failure();
      for (const auto &successor : outgoing.edges)
        append(successor.target, change);
    }
    return success();
  }

  template <typename Op>
  LogicalResult propagateStructuredOperand(Op operation, OpOperand &use,
                                           const ExtentChange &change) {
    auto structured = cast<StructuredOpInterface>(operation.getOperation());
    unsigned operand = use.getOperandNumber();
    auto sources = operation.getSources();
    auto captures = operation.getCaptures();
    if (!captures.empty() && operand >= captures.getBeginOperandIndex() &&
        operand < captures.getBeginOperandIndex() + captures.size()) {
      unsigned component = operand - captures.getBeginOperandIndex();
      if (structured.getSummarizeRegion()) {
        append(structured.getSummarizeCaptures()[component], change);
        if (structured.getEmitRegion()) append(structured.getEmitCaptures()[component], change);
      } else {
        append(structured.getCombineCaptures()[component], change);
      }
      return success();
    }
    if (operand < sources.getBeginOperandIndex() ||
        operand >= sources.getBeginOperandIndex() + sources.size())
      return success();
    unsigned component = operand - sources.getBeginOperandIndex();
    auto axes = structured.getIterationAxes();
    bool member = llvm::is_contained(axes, static_cast<int64_t>(change.occurrence.axis));
    if (structured.getSummarizeRegion()) {
      if (!member) {
        append(structured.getSummarizeSources()[component], change);
        if (structured.getEmitRegion()) append(structured.getEmitSources()[component], change);
      }
    } else if (structured.getStructuredKind() == StructuredOpKind::Scan || !member) {
      ExtentChange result = change;
      if (structured.getStructuredKind() == StructuredOpKind::Reduce)
        result.occurrence.axis -= llvm::count_if(axes, [&](int64_t axis) {
          return axis < static_cast<int64_t>(change.occurrence.axis);
        });
      appendCollectiveComponent(structured, component, result);
    }
    return success();
  }

  LogicalResult propagateStructuredYield(const ExtentChange &change,
                                         OpOperand &use) {
    auto structured = dyn_cast<StructuredOpInterface>(use.getOwner()->getParentOp());
    if (!structured)
      return success();
    if (isa<ReduceOp, ScanOp>(structured.getOperation())) {
      appendCollectiveComponent(structured, use.getOperandNumber(), change);
      return success();
    }
    for (const auto &group : queryStructuredSchemaGroups(structured))
      if (group.producer == &use || llvm::is_contained(group.yields, &use))
        appendGroup(group, change);
    if (auto scan = dyn_cast<RegionScanOp>(structured.getOperation());
        scan && use.getOwner()->getParentRegion() == &scan.getEmit())
      return propagateEmission(scan, use.getOperandNumber(),
                                scan.getEmittedResults()[use.getOperandNumber()], change);
    return success();
  }

  LogicalResult propagateAccess(const ExtentChange &change, OpOperand &use,
                                 AccessOpInterface access) {
    if (!change.occurrence.fields.empty())
      return success();
    auto result = access.getAccessResult();
    if (!result)
      return success();
    if (access.getAccessKind() == AccessKind::Gather &&
        &use == &access.getAccessResourceOperand()) {
      if (llvm::is_contained(access.getAccessSourceAxes(),
                             static_cast<int64_t>(change.occurrence.axis)))
        return success();
      const auto &snapshot = snapshots.find(use.getOwner())->second;
      auto source = cast<FragmentType>(snapshot.operands[use.getOperandNumber()]);
      auto retained = inferCollectiveResultType(
          source, access.getAccessSourceAxes(), source.getElementType());
      auto target = dyn_cast<FragmentType>(snapshot.resultTypes.front());
      if (failed(retained) || !isa<FragmentType>(*retained) || !target)
        return access.emitOpError("gather retained axes have no fragment relation");
      auto projection = queryAxisProjection(cast<FragmentType>(*retained), target);
      if (!projection.isExact())
        return access.emitOpError("gather retained axes have no unique result projection");
      unsigned retainedAxis = change.occurrence.axis - llvm::count_if(
          access.getAccessSourceAxes(), [&](int64_t axis) {
            return axis < static_cast<int64_t>(change.occurrence.axis);
          });
      for (auto [axis, from] : llvm::enumerate(projection.targetToSource))
        if (from && *from == retainedAxis)
          append(result, {{}, static_cast<unsigned>(axis)}, change.extent);
      return success();
    }
    auto coordinates = access.getAccessCoordinates();
    unsigned operand = use.getOperandNumber();
    if (!coordinates.empty() && operand >= coordinates.getBeginOperandIndex() &&
        operand < coordinates.getBeginOperandIndex() + coordinates.size()) {
      unsigned slot = operand - coordinates.getBeginOperandIndex();
      const auto &projection = snapshots.find(use.getOwner())->second.coordinates[slot];
      if (projection.state == BroadcastProjectionState::Ambiguous)
        return access.emitOpError("extent decision has an ambiguous access coordinate slot");
      for (auto [axis, source] : llvm::enumerate(projection.targetToSource))
        if (source && *source == change.occurrence.axis)
          append(result, {{}, static_cast<unsigned>(axis)}, change.extent);
      return success();
    }
    auto payloads = access.getAccessPayloads();
    if (!payloads.empty() && operand >= payloads.getBeginOperandIndex() &&
        operand < payloads.getBeginOperandIndex() + payloads.size())
      append(result, change);
    return success();
  }

  template <typename Op>
  void propagateContract(Op contract, OpOperand &lhs, OpOperand &rhs, OpOperand &use,
                         const ExtentChange &change) {
    if (!change.occurrence.fields.empty())
      return;
    if (&use == &contract.getAccumulatorMutable()) {
      append(contract.getResult(), change);
      return;
    }
    auto axes = queryContractionAxes(contract);
    if (!axes)
      return;
    const auto *positions = &use == &lhs ? &axes->lhsResultAxes
                            : &use == &rhs ? &axes->rhsResultAxes : nullptr;
    if (positions && change.occurrence.axis < positions->size() &&
        (*positions)[change.occurrence.axis])
      append(contract.getResult(), {{}, *(*positions)[change.occurrence.axis]}, change.extent);
  }

  void resizeInitialStorage(BufferOp buffer, const ExtentChange &change,
                            Type previous) {
    if (buffer.getInitialValue() != change.value || !change.occurrence.fields.empty())
      return;
    auto before = dyn_cast<FragmentType>(previous);
    auto after = cast<FragmentType>(change.value.getType());
    auto storage = buffer.getResult().getType();
    if (!before || storage.getShape() != before.getShape() ||
        storage.getOwner() != before.getOwner() || after.getOwner() != storage.getOwner())
      return;
    bool padding = llvm::all_of(llvm::zip(before.getShape(), after.getShape()), [](auto pair) {
      auto [left, right] = pair;
      if (left == right) return true;
      auto oldExtent = cast<PhysicalExprAttr>(left);
      auto newExtent = cast<PhysicalExprAttr>(right);
      return oldExtent.getKind() == PhysicalExprKind::Constant &&
             newExtent.getKind() == PhysicalExprKind::Constant && oldExtent.getValue() > 0 &&
             static_cast<uint64_t>(newExtent.getValue()) == llvm::PowerOf2Ceil(static_cast<uint64_t>(oldExtent.getValue()));
    });
    if (padding)
      setPhysicalValueType(buffer.getResult(), BufferType::get(
          storage.getContext(), storage.getElementType(), after.getShape(),
          storage.getScope(), storage.getInstance(), storage.getOwner(),
          storage.getInitialization(), storage.getVisibility()), changed);
  }

  LogicalResult propagateUse(const ExtentChange &change, OpOperand &use,
                             Type previous) {
    Operation *owner = use.getOwner();
    if (auto record = dyn_cast<MakeRecordOp>(owner)) {
      AxisOccurrence occurrence = change.occurrence;
      occurrence.fields.insert(occurrence.fields.begin(), use.getOperandNumber());
      append(record.getResult(), occurrence, change.extent);
    } else if (auto extract = dyn_cast<ExtractOp>(owner)) {
      if (!change.occurrence.fields.empty() &&
          change.occurrence.fields.front() == extract.getField()) {
        AxisOccurrence occurrence = change.occurrence;
        occurrence.fields.erase(occurrence.fields.begin());
        append(extract.getResult(), occurrence, change.extent);
      }
    } else if (isa<FragmentOpInterface>(owner)) {
      return propagateFragmentUse(change, use);
    } else if (auto access = dyn_cast<AccessOpInterface>(owner)) {
      return propagateAccess(change, use, access);
    } else if (auto op = dyn_cast<ReduceOp>(owner)) {
      return propagateStructuredOperand(op, use, change);
    } else if (auto op = dyn_cast<ScanOp>(owner)) {
      return propagateStructuredOperand(op, use, change);
    } else if (auto op = dyn_cast<RegionFoldOp>(owner)) {
      return propagateStructuredOperand(op, use, change);
    } else if (auto op = dyn_cast<RegionScanOp>(owner)) {
      return propagateStructuredOperand(op, use, change);
    } else if (isa<YieldOp>(owner)) {
      return propagateStructuredYield(change, use);
    } else if (isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
                   BranchOpInterface>(owner)) {
      auto outgoing = queryControlFlowOutgoing(use);
      if (!outgoing.complete)
        return owner->emitOpError("cannot retarget an unknown control successor slot");
      for (const auto &edge : outgoing.edges) append(edge.target, change);
    } else if (auto op = dyn_cast<ContractOp>(owner)) {
      propagateContract(op, op.getLhsMutable(), op.getRhsMutable(), use, change);
    } else if (auto op = dyn_cast<ScaledContractOp>(owner)) {
      propagateContract(op, op.getLhsMutable(), op.getRhsMutable(), use, change);
    } else if (auto op = dyn_cast<SparseContractOp>(owner)) {
      propagateContract(op, op.getCompressedMutable(), op.getRhsMutable(), use, change);
    } else if (auto buffer = dyn_cast<BufferOp>(owner)) {
      resizeInitialStorage(buffer, change, previous);
    }
    return success();
  }

  Value root;
  ValueTypeChangeCallback changed;
  OpBuilder::Listener *listener;
  SmallVector<ExtentChange> pending;
  DenseMap<Value, SmallVector<Visit>> visited;
  DenseMap<Operation *, Snapshot> snapshots;
  llvm::SetVector<Value> boundaries;
};

LogicalResult locateAxes(Type type, llvm::function_ref<bool(AxisMapAttr)> selects,
                         SmallVectorImpl<AxisOccurrence> &selected,
                         SmallVector<unsigned> fields = {}) {
  if (auto record = dyn_cast<RecordType>(type)) {
    for (auto [index, field] : llvm::enumerate(record.getFieldTypes())) {
      auto path = fields;
      path.push_back(index);
      if (failed(locateAxes(cast<TypeAttr>(field).getValue(), selects, selected, path)))
        return failure();
    }
    return success();
  }
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return success();
  std::optional<unsigned> selectedAxis;
  for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
    if (!selects(cast<AxisMapAttr>(attribute)))
      continue;
    if (selectedAxis)
      return failure();
    selectedAxis = axis;
  }
  if (selectedAxis)
    selected.push_back({std::move(fields), *selectedAxis});
  return success();
}

LogicalResult retargetLocatedExtent(Value root,
                                    llvm::function_ref<bool(AxisMapAttr)> selects,
                                    PhysicalExprAttr extent,
                                    ValueTypeChangeCallback changed,
                                    OpBuilder::Listener *listener) {
  SmallVector<AxisOccurrence> selected;
  if (failed(locateAxes(root.getType(), selects, selected)))
    return emitError(root.getLoc(), "extent root has multiple matching fragment axes; select an explicit occurrence");
  return ExtentPropagation(root, changed, listener).run(selected, extent);
}

} // namespace

LogicalResult retargetFragmentAxisExtent(Value root, unsigned fragmentAxis,
                                         PhysicalExprAttr extent,
                                         ValueTypeChangeCallback changed,
                                         OpBuilder::Listener *listener) {
  auto fragment = dyn_cast<FragmentType>(root.getType());
  if (!fragment || fragmentAxis >= fragment.getShape().size())
    return emitError(root.getLoc(), "extent decision requires an existing fragment axis");
  AxisOccurrence selected{{}, fragmentAxis};
  return ExtentPropagation(root, changed, listener).run({selected}, extent);
}

LogicalResult retargetValueExtents(Value root, Type selected,
                                   ValueTypeChangeCallback changed,
                                   OpBuilder::Listener *listener) {
  SmallVector<ExtentChange> changes;
  std::function<LogicalResult(Type, Type, SmallVector<unsigned>)> collect =
      [&](Type current, Type target, SmallVector<unsigned> fields) -> LogicalResult {
    if (current == target)
      return success();
    if (auto record = dyn_cast<RecordType>(current)) {
      auto destination = dyn_cast<RecordType>(target);
      if (!destination || record.getFieldNames() != destination.getFieldNames() ||
          record.getFieldTypes().size() != destination.getFieldTypes().size() ||
          record.getOwner() != destination.getOwner())
        return failure();
      for (auto [index, field] : llvm::enumerate(record.getFieldTypes())) {
        auto path = fields;
        path.push_back(index);
        if (failed(collect(cast<TypeAttr>(field).getValue(),
                cast<TypeAttr>(destination.getFieldTypes()[index]).getValue(),
                std::move(path))))
          return failure();
      }
      return success();
    }
    auto source = dyn_cast<FragmentType>(current);
    auto destination = dyn_cast<FragmentType>(target);
    if (!source || !destination ||
        source.getElementType() != destination.getElementType() ||
        source.getShape().size() != destination.getShape().size() ||
        source.getAxisMaps() != destination.getAxisMaps() ||
        source.getValidity() != destination.getValidity() ||
        source.getOwner() != destination.getOwner())
      return failure();
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
      if (source.getShape()[axis] != destination.getShape()[axis])
        changes.push_back({root, {fields, axis},
            cast<PhysicalExprAttr>(destination.getShape()[axis])});
    return success();
  };
  if (failed(collect(root.getType(), selected, {})))
    return emitError(root.getLoc(),
                     "selected extent schema changes a field or coordinate relation");
  return ExtentPropagation(root, changed, listener).run(changes);
}

LogicalResult retargetSourceExtent(Value root, PhysicalSourceAxis source,
                                   PhysicalExprAttr extent,
                                   std::optional<int64_t> dimension,
                                   ValueTypeChangeCallback changed,
                                   OpBuilder::Listener *listener) {
  return retargetLocatedExtent(root, [=](AxisMapAttr mapping) {
    return sourceAxisIdentity(mapping) == source &&
           (!dimension || mapping.getDimensionId() == *dimension);
  }, extent, changed, listener);
}

LogicalResult retargetDimensionExtent(Value root, int64_t dimensionId,
                                      PhysicalExprAttr extent,
                                      ValueTypeChangeCallback changed,
                                      OpBuilder::Listener *listener) {
  if (dimensionId <= 0)
    return success();
  return retargetLocatedExtent(root, [=](AxisMapAttr mapping) {
    return mapping.getDimensionId() == dimensionId;
  }, extent, changed, listener);
}

} // namespace intent::gpu
