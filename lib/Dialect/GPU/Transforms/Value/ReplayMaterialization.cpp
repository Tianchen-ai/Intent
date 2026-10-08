#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "ReplayInsertion.h"
#include "ReplayPolicy.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;

namespace intent::gpu {
namespace {

// A source-slice replay owns its seeds, dominance scope and invariant-value
// memoization. Only identical root selections reuse an invariant result.
class RangeValueReplay {
public:
  RangeValueReplay(OpBuilder &builder, Location location, func::FuncOp kernel,
                   PhysicalExprAttr blockedExtent, Value replacement,
                   IRMapping &mapping, Operation *insertionAnchor)
      : builder(builder), location(location), kernel(kernel),
        blockedExtent(blockedExtent), replacement(replacement),
        mapping(mapping), insertionAnchor(insertionAnchor), dominance(kernel) {}

  FailureOr<Value> replay(Value value, ArrayRef<MakeRangeOp> roots) {
  if (Value mapped = mapping.lookupOrNull(value))
    return availableAtInsertionPoint(mapped, builder, dominance)
        ? FailureOr<Value>(mapped) : FailureOr<Value>(failure());
  if (isInvariant(value, roots))
    return value;
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      Value field = record.getFields()[extract.getField()];
      FailureOr<Value> replayed = replay(field, roots);
      if (failed(replayed))
        return failure();
      mapping.map(value, *replayed);
      return *replayed;
    }
  }
  if (auto range = value.getDefiningOp<MakeRangeOp>())
    if (llvm::any_of(roots, [&](MakeRangeOp root) {
          return range == root || sameLogicalRange(range, root);
        })) {
      auto original = range.getResult().getType();
      auto coordinate = cast<FragmentType>(replacement.getType());
      auto target = FragmentType::get(
          value.getContext(), original.getElementType(), coordinate.getShape(),
          original.getAxisMaps(), original.getValidity(), original.getOwner());
      return projectPhysicalValueToSchema(builder, location, replacement, target);
    }
  bool relocate = !availableAtInsertionPoint(value, builder, dominance);
  auto originalResultType = dyn_cast<FragmentType>(value.getType());
  if (!originalResultType) {
    if (!relocate)
      return rememberInvariant(value, roots);
    Operation *producer = value.getDefiningOp();
    if (!producer || producer->getNumRegions() != 0 ||
        producer->getNumResults() != 1 ||
        (!isa<arith::ConstantOp>(producer) &&
         !isPhysicalReplayNode(producer, PhysicalReplayScope::ValueGraph,
                               /*allowAccesses=*/true)))
      return failure();
    for (Value operand : producer->getOperands()) {
      FailureOr<Value> replayed = replay(operand, roots);
      if (failed(replayed))
        return failure();
      if (*replayed != operand && !mapping.lookupOrNull(operand))
        mapping.map(operand, *replayed);
    }
    if (auto load = dyn_cast<LoadOp>(producer);
        load && !canReplayReadAt(load, insertionAnchor))
      return failure();
    auto cloned = cloneWithSchema(builder, producer, mapping, producer->getResultTypes());
    if (failed(cloned)) return failure();
    return (*cloned)[0];
  }
  if (!kernel)
    return failure();
  PhysicalRangeAxisFact selected =
      PhysicalProgramAnalysis(kernel).rangeAxes(value, roots);
  if (!selected.isExact()) {
    InFlightDiagnostic diagnostic =
        value.getDefiningOp()
            ? value.getDefiningOp()->emitOpError(
                  "selected range roots have no exact fragment-axis projection")
            : kernel.emitError(
                  "selected range roots have no exact fragment-axis projection");
    for (Operation *blocker : selected.blockers)
      diagnostic << "; blocker=" << blocker->getName();
    diagnostic << "; value_type=" << value.getType();
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
      diagnostic << "; broadcast_input_type=" << broadcast.getValue().getType();
    return failure();
  }
  if (selected.fragmentAxes.empty() && !relocate)
    return rememberInvariant(value, roots);
  Operation *producer = value.getDefiningOp();
  bool retainedBoundary = isa<BlockArgument>(value);
  if (!retainedBoundary && producer && producer->getNumRegions() &&
      !relocate && selected.fragmentAxes.size() == 1) {
    unsigned axis = selected.fragmentAxes.front();
    PhysicalProgramAnalysis analysis(kernel);
    auto authority = completeSnapshotRange(value, axis, analysis);
    if (succeeded(authority) &&
        availableAtInsertionPoint((*authority).getStart(), builder, dominance) &&
        availableAtInsertionPoint((*authority).getExtent(), builder, dominance) &&
        availableAtInsertionPoint((*authority).getLogicalStop(), builder, dominance))
      retainedBoundary = true;
  }
  if (retainedBoundary && !relocate &&
      selected.fragmentAxes.size() == 1 &&
      PhysicalProgramAnalysis(kernel)
          .axisRealization(value, selected.fragmentAxes.front()).physicalized) {
    // A dominating argument or completed region result is an existing SSA
    // snapshot. Slice its actual lanes rather than replaying ordered state.
    // Region results above cover their complete selected physical axis, so
    // slicing cannot replace an existing padded lane with the helper's fill.
    auto slice = materializeRetainedSlice(
        builder, location, value, selected.fragmentAxes.front(), blockedExtent,
        replacement, insertionAnchor);
    if (failed(slice))
      return failure();
    mapping.map(value, *slice);
    return *slice;
  }
  if (!producer || (isa<MakeRangeOp>(producer) && !relocate)) {
    if (producer)
      producer->emitOpError(
          "selected range root was not bound to its blocked replacement");
    return failure();
  }
  if (!isa<MakeRangeOp>(producer) &&
      !isPhysicalReplayNode(producer, PhysicalReplayScope::ValueGraph,
                            /*allowAccesses=*/true)) {
    producer->emitOpError(
        "selected range value is not a replayable physical value node");
    return failure();
  }
  if (producer->getNumRegions() != 0 || producer->getNumResults() != 1) {
    producer->emitOpError(
        "selected range value does not have a single replayable result");
    return failure();
  }
  SmallVector<MakeRangeOp> operandRoots(roots.begin(), roots.end());
  for (unsigned axis : selected.fragmentAxes)
    for (MakeRangeOp root :
         PhysicalProgramAnalysis(kernel).axisRanges(value, axis).roots)
      if (!llvm::is_contained(operandRoots, root))
        operandRoots.push_back(root);
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replayed = replay(operand, operandRoots);
    if (failed(replayed)) {
      producer->emitOpError(
          "selected range value has an operand that cannot be replayed");
      return failure();
    }
    if (*replayed != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replayed);
  }
  SmallVector<Attribute> targetShape(originalResultType.getShape().begin(),
                                     originalResultType.getShape().end());
  for (unsigned axis : selected.fragmentAxes)
    targetShape[axis] = blockedExtent;
  FragmentType targetType = FragmentType::get(
      originalResultType.getContext(), originalResultType.getElementType(),
      ArrayAttr::get(originalResultType.getContext(), targetShape),
      originalResultType.getAxisMaps(), originalResultType.getValidity(),
      originalResultType.getOwner());
  if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
    PhysicalRangeAxisFact input =
        PhysicalProgramAnalysis(kernel).rangeAxes(reshape.getValue(), roots);
    if (!input.isExact()) {
      reshape.emitOpError(
          "reshape input has no exact selected-range axis projection");
      return failure();
    }
    if (input.fragmentAxes.empty()) {
      Value input = mapping.lookupOrDefault(reshape.getValue());
      auto broadcast = builder.create<BroadcastOp>(location, targetType, input);
      if (!mapping.lookupOrNull(value))
        mapping.map(value, broadcast.getResult());
      return broadcast.getResult();
    }
  }
  IRMapping cloneMapping(mapping);
  if (auto load = dyn_cast<LoadOp>(producer);
      load && !canReplayReadAt(load, insertionAnchor))
    return failure();
  auto cloned = cloneWithSchema(builder, producer, cloneMapping, TypeRange{targetType});
  if (failed(cloned)) return failure();
  if (!mapping.lookupOrNull(value))
    mapping.map(value, (*cloned)[0]);
  return (*cloned)[0];

  }

private:
  bool isInvariant(Value value, ArrayRef<MakeRangeOp> roots) const {
    auto found = invariantValues.find(value);
    return found != invariantValues.end() &&
           llvm::any_of(found->second, [&](const auto &selected) {
             return ArrayRef<MakeRangeOp>(selected) == roots;
           });
  }

  Value rememberInvariant(Value value, ArrayRef<MakeRangeOp> roots) {
    invariantValues[value].emplace_back(roots.begin(), roots.end());
    return value;
  }

  OpBuilder &builder;
  Location location;
  func::FuncOp kernel;
  PhysicalExprAttr blockedExtent;
  Value replacement;
  IRMapping &mapping;
  Operation *insertionAnchor;
  DominanceInfo dominance;
  llvm::DenseMap<Value, SmallVector<SmallVector<MakeRangeOp>>> invariantValues;
};

} // namespace

FailureOr<Value> materializeReplayedRanges(
    OpBuilder &builder, Location location, Value value,
    PhysicalExprAttr blockedExtent, ArrayRef<MakeRangeOp> roots,
    Value replacement, IRMapping &resultMapping, Operation *insertionAnchor) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel || roots.empty() || !insertionAnchor ||
      !builder.getInsertionBlock())
    return failure();
  IRMapping mapping(resultMapping);
  ReplayInsertion insertion(builder);
  auto source = sourceAxisIdentity(roots.front());
  auto dimension = queryRangeDimension(roots.front());
  if (failed(dimension) || !llvm::all_of(roots, [&](MakeRangeOp root) {
        auto current = queryRangeDimension(root);
        return sourceAxisIdentity(root) == source && succeeded(current) &&
               *current == *dimension;
      }))
    return failure();
  auto proof = PhysicalProgramAnalysis(kernel).replayAt(
      value, source, PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true,
      insertionAnchor, mapping, *dimension);
  if (!proof.isReplayable())
    return failure();
  DominanceInfo dominance(kernel);
  if (!availableAtInsertionPoint(replacement, builder, dominance))
    return failure();
  RangeValueReplay materialization(builder, location, kernel, blockedExtent,
                                   replacement, mapping, insertionAnchor);
  auto result = materialization.replay(value, roots);
  if (failed(result)) return failure();
  resultMapping = std::move(mapping);
  insertion.commit();
  return result;
}

FailureOr<Value> materializeReplayedValue(
    OpBuilder &builder, Location location, Value value,
    PhysicalSourceAxis source, PhysicalExprAttr blockedExtent,
    IRMapping &resultMapping, Operation *insertionAnchor,
    ReplayMaterializationOptions options) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel || !insertionAnchor || !builder.getInsertionBlock())
    return failure();
  IRMapping mapping(resultMapping);
  ReplayInsertion insertion(builder);
  PhysicalProgramAnalysis analysis(kernel);
  DominanceInfo dominance(kernel);
  auto available = [&](Value current) {
    return availableAtInsertionPoint(current, builder, dominance);
  };
  auto replayDimension = [&](PhysicalSourceAxis occurrence)
      -> std::optional<int64_t> {
    auto projections = queryFragmentAxes(value.getType(), occurrence);
    if (options.fragmentAxis)
      llvm::erase_if(projections, [&](PhysicalAxisProjection projection) {
        return projection.fragmentAxis != *options.fragmentAxis;
      });
    return projections.size() == 1
               ? std::optional<int64_t>(projections.front().dimensionId)
               : std::nullopt;
  };
  PhysicalReplayFact replay = analysis.replayAt(
      value, source, options.scope, options.allowAccesses, insertionAnchor, mapping,
      replayDimension(source));
  if (!replay.isReplayable())
    return failure();

  SmallVector<PhysicalSourceAxis> replaySources{source};
  for (MakeRangeOp range : options.traversalRanges) {
    PhysicalSourceAxis occurrence = sourceAxisIdentity(range);
    if (!llvm::is_contained(replaySources, occurrence))
      replaySources.push_back(occurrence);
    if (!analysis.replayAt(value, occurrence, options.scope,
                               options.allowAccesses, insertionAnchor, mapping,
                               replayDimension(occurrence)).isReplayable())
      return failure();
  }
  bool projectionFailed = false;
  auto replayProjection = [&](Type type,
                              std::optional<unsigned> axis = std::nullopt) {
    PhysicalAxisProjection result;
    for (PhysicalSourceAxis occurrence : replaySources) {
      auto projections = queryFragmentAxes(type, occurrence);
      auto dimension = replayDimension(occurrence);
      llvm::erase_if(projections, [&](PhysicalAxisProjection projection) {
        return (axis && projection.fragmentAxis != *axis) ||
               (dimension && projection.dimensionId != *dimension);
      });
      PhysicalAxisProjection current;
      if (projections.size() == 1)
        current = projections.front();
      if (projections.size() > 1 ||
          (result.isExact() && current.isExact() &&
           (result.fragmentAxis != current.fragmentAxis ||
            result.dimensionId != current.dimensionId))) {
        result.state = PhysicalFactState::Ambiguous;
        projectionFailed = true;
        return result;
      }
      if (!current.isExact())
        continue;
      result = current;
    }
    return result;
  };
  auto replaceReplayAxis = [&](FragmentType type, unsigned axis) {
    SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
    SmallVector<Attribute> axes(type.getAxisMaps().begin(),
                                type.getAxisMaps().end());
    shape[axis] = blockedExtent;
    if (options.segmentMapping)
      axes[axis] = AxisMapAttr::get(
          type.getContext(), options.segmentMapping.getSourceId(),
          options.segmentMapping.getSourceAxis(),
          options.segmentMapping.getDimensionId(), axis,
          options.segmentMapping.getDerived());
    return FragmentType::get(
        type.getContext(), type.getElementType(),
        ArrayAttr::get(type.getContext(), shape),
        ArrayAttr::get(type.getContext(), axes), type.getValidity(),
        type.getOwner());
  };
  std::function<Type(Type)> replaceReplayType = [&](Type type) -> Type {
    if (auto fragment = dyn_cast<FragmentType>(type)) {
      PhysicalAxisProjection projection = replayProjection(fragment);
      return projection.isExact()
                 ? Type(replaceReplayAxis(fragment, projection.fragmentAxis))
                 : type;
    }
    auto record = dyn_cast<RecordType>(type);
    if (!record)
      return type;
    SmallVector<Attribute> fields;
    bool changed = false;
    for (Attribute field : record.getFieldTypes()) {
      Type current = cast<TypeAttr>(field).getValue();
      Type replacement = replaceReplayType(current);
      fields.push_back(TypeAttr::get(replacement));
      changed |= replacement != current;
    }
    return changed ? Type(RecordType::get(
                         type.getContext(), record.getFieldNames(),
                         ArrayAttr::get(type.getContext(), fields),
                         record.getOwner()))
                   : type;
  };
  auto carriesReplaySource = [&](Type type) {
    return replaceReplayType(type) != type;
  };
  auto selectedReplayAxis = [&](Value current,
                                std::optional<unsigned> axis = std::nullopt)
      -> std::optional<unsigned> {
    auto fragment = dyn_cast<FragmentType>(current.getType());
    PhysicalAxisProjection projection =
        fragment ? replayProjection(fragment, axis)
                 : PhysicalAxisProjection{};
    if (fragment && !projection.isExact() &&
        projection.state != PhysicalFactState::Ambiguous &&
        !options.traversalRanges.empty()) {
      std::optional<unsigned> selected;
      for (unsigned candidate = 0; candidate < fragment.getShape().size(); ++candidate) {
        if (axis && candidate != *axis)
          continue;
        // Equal traversal bounds do not identify independent logical axes.
        auto candidateMap = cast<AxisMapAttr>(fragment.getAxisMaps()[candidate]);
        if (!llvm::any_of(options.traversalRanges, [&](MakeRangeOp range) {
              auto dimension = queryRangeDimension(range);
              return succeeded(dimension) &&
                     *dimension == candidateMap.getDimensionId();
            }))
          continue;
        PhysicalRangeFact fact = analysis.axisRanges(current, candidate);
        if (fact.state == PhysicalFactState::Unknown || !fact.blockers.empty() ||
            fact.roots.empty())
          continue;
        if (!llvm::any_of(fact.roots, [&](MakeRangeOp root) {
              return llvm::any_of(options.traversalRanges, [&](MakeRangeOp range) {
                return sameLogicalRange(root, range);
              });
            }))
          continue;
        SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
        combined.append(options.traversalRanges.begin(), options.traversalRanges.end());
        if (!analysis.lockstepRanges(combined).isExact())
          continue;
        if (selected)
          return std::nullopt;
        selected = candidate;
      }
      if (selected)
        return selected;
    }
    if (!fragment || !projection.isExact())
      return std::nullopt;
    if (options.traversalRanges.empty())
      return projection.fragmentAxis;
    PhysicalRangeFact fact =
        analysis.axisRanges(current, projection.fragmentAxis);
    if (fact.roots.empty()) {
      Operation *producer = current.getDefiningOp();
      bool neutralSchemaCarrier = isa_and_nonnull<SplatOp>(producer);
      neutralSchemaCarrier |= fact.isExact() && fact.blockers.empty() &&
          isa_and_nonnull<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
                         BitcastOp>(producer);
      if (auto broadcast = dyn_cast_or_null<BroadcastOp>(producer)) {
        auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
        neutralSchemaCarrier |=
            !isa<FragmentType, RecordType>(broadcast.getValue().getType());
        if (input) {
          BroadcastProjection relation = queryAxisProjection(input, fragment);
          if (relation.isExact()) {
            std::optional<unsigned> inputAxis =
                relation.targetToSource[projection.fragmentAxis];
            if (!inputAxis) {
              neutralSchemaCarrier = true;
            } else {
              auto extent = cast<PhysicalExprAttr>(input.getShape()[*inputAxis]);
              neutralSchemaCarrier |=
                  extent.getKind() ==
                      PhysicalExprKind::Constant &&
                  extent.getValue() == 1;
            }
          }
        }
      }
      return neutralSchemaCarrier
                 ? std::optional<unsigned>(projection.fragmentAxis)
                 : std::nullopt;
    }
    SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
    combined.append(options.traversalRanges.begin(),
                    options.traversalRanges.end());
    return analysis.lockstepRanges(combined).isExact()
               ? std::optional<unsigned>(projection.fragmentAxis)
               : std::nullopt;
  };
  auto replayHelperType = [&](Value current, PhysicalExprAttr logicalExtent) -> Type {
    auto fragment = dyn_cast<FragmentType>(current.getType());
    PhysicalAxisProjection projection =
        fragment ? queryFragmentAxis(fragment, source, replayDimension(source))
                 : PhysicalAxisProjection{};
    if (!fragment || !projection.isExact() ||
        fragment.getShape()[projection.fragmentAxis] != logicalExtent)
      return current.getType();
    return replaceReplayAxis(fragment, projection.fragmentAxis);
  };

  llvm::DenseMap<std::pair<Value, unsigned>, Value> axisValues;
  auto hasMultipleReplayAxes = [&](Type type) {
    std::optional<unsigned> selected;
    for (PhysicalSourceAxis occurrence : replaySources)
      for (PhysicalAxisProjection projection :
           queryFragmentAxes(type, occurrence)) {
        if (selected && *selected != projection.fragmentAxis)
          return true;
        selected = projection.fragmentAxis;
      }
    return false;
  };
  auto remember = [&](Value original, Value replacement,
                       std::optional<unsigned> axis) {
    if (axis)
      axisValues[{original, *axis}] = replacement;
    if (!hasMultipleReplayAxes(original.getType()))
      mapping.map(original, replacement);
  };
  std::function<FailureOr<Value>(Value, std::optional<unsigned>)> materialize =
      [&](Value current,
          std::optional<unsigned> requestedAxis) -> FailureOr<Value> {
    auto fragment = dyn_cast<FragmentType>(current.getType());
    std::optional<unsigned> projection =
        selectedReplayAxis(current, requestedAxis);
    if (projection) {
      auto cached = axisValues.find({current, *projection});
      if (cached != axisValues.end())
        return cached->second;
    }
    if (Value mapped = mapping.lookupOrNull(current)) {
      if (!available(mapped))
        return failure();
      // A prebound load must identify its occurrence by the changed axis alone.
      if (hasMultipleReplayAxes(current.getType()) &&
          (!projection || fragment.getShape()[*projection] == blockedExtent ||
           mapped.getType() != replaceReplayAxis(fragment, *projection)))
        return failure();
      return mapped;
    }
    if (auto extract = current.getDefiningOp<ExtractOp>()) {
      if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
        uint64_t field = extract.getField();
        if (field >= record.getFields().size())
          return failure();
        FailureOr<Value> replayed =
            materialize(record.getFields()[field], projection);
        if (succeeded(replayed))
          remember(current, *replayed, projection);
        return replayed;
      }
      if (carriesReplaySource(extract.getRecord().getType())) {
        FailureOr<Value> replayedRecord =
            materialize(extract.getRecord(), std::nullopt);
        if (failed(replayedRecord))
          return failure();
        Type target = replaceReplayType(current.getType());
        Value replayed = builder.create<ExtractOp>(
            location, target, *replayedRecord, extract.getField());
        remember(current, replayed, projection);
        return replayed;
      }
    }

    if (!fragment && carriesReplaySource(current.getType())) {
      Operation *producer = current.getDefiningOp();
      if (!producer ||
          !isPhysicalReplayNode(producer, options.scope,
                                /*allowAccesses=*/false))
        return failure();
      IRMapping cloneMapping(mapping);
      for (Value operand : producer->getOperands()) {
        if (!mapping.lookupOrNull(operand) &&
            !carriesReplaySource(operand.getType()) && available(operand)) {
          cloneMapping.map(operand, operand);
          continue;
        }
        FailureOr<Value> replayed = materialize(operand, std::nullopt);
        if (failed(replayed))
          return failure();
        cloneMapping.map(operand, *replayed);
      }
      auto cloned = cloneWithPhysicalSchema(builder, producer, cloneMapping, [&](Value original) {
            return replaceReplayType(original.getType());
          });
      if (failed(cloned)) return failure();
      for (auto [original, result] :
           llvm::zip(producer->getResults(), *cloned)) {
        if (!mapping.lookupOrNull(original))
          mapping.map(original, result);
      }
      auto result = dyn_cast<OpResult>(current);
      if (!result || result.getResultNumber() >= cloned->size())
        return failure();
      Value replayed = (*cloned)[result.getResultNumber()];
      return replayed;
    }
    if (!fragment) {
      if (available(current))
        return current;
      Operation *producer = current.getDefiningOp();
      if (!producer || producer->getNumRegions() != 0 ||
          producer->getNumResults() != 1 ||
          (!isa<arith::ConstantOp>(producer) &&
           !isPhysicalReplayNode(producer, options.scope, /*allowAccesses=*/false)))
        return failure();
      IRMapping operands(mapping);
      for (Value operand : producer->getOperands()) {
        auto replayed = materialize(operand, std::nullopt);
        if (failed(replayed))
          return failure();
        operands.map(operand, *replayed);
      }
      auto cloned = cloneWithSchema(builder, producer, operands, producer->getResultTypes());
      if (failed(cloned)) return failure();
      mapping.map(current, (*cloned)[0]);
      return (*cloned)[0];
    }
    if (!projection) {
      Operation *producer = current.getDefiningOp();
      if (!producer || !isPhysicalReplayNode(
                           producer, PhysicalReplayScope::Coordinate,
                           /*allowAccesses=*/false))
        return available(current) ? FailureOr<Value>(current)
                                  : FailureOr<Value>(failure());
      IRMapping cloneMapping(mapping);
      bool changed = false;
      for (Value operand : producer->getOperands()) {
        FailureOr<Value> replayed = materialize(operand, std::nullopt);
        if (failed(replayed))
          return failure();
        cloneMapping.map(operand, *replayed);
        changed |= *replayed != operand;
      }
      if (!changed && available(current))
        return current;
      auto cloned = cloneWithPhysicalSchema(builder, producer, cloneMapping,
          [](Value original) { return original.getType(); });
      if (failed(cloned)) return failure();
      remember(current, (*cloned)[0], std::nullopt);
      return (*cloned)[0];
    }
    unsigned axis = *projection;
    Operation *producer = current.getDefiningOp();
    if (!producer)
      return failure();

    auto operandRelations = queryFragmentOperandRelations(producer);
    auto replayOperand = [&](Value operand) -> FailureOr<Value> {
      std::optional<unsigned> operandAxis;
      auto input = dyn_cast<FragmentType>(operand.getType());
      if (input && (hasMultipleReplayAxes(input) ||
                    hasMultipleReplayAxes(fragment))) {
        auto reduction = dyn_cast<ReduceOp>(producer);
        bool reductionSource = reduction && llvm::is_contained(
            reduction.getSources(), operand);
        if (isa<FragmentOpInterface>(producer)) {
          if (failed(operandRelations))
            return failure();
          bool found = false;
          bool introduced = false;
          for (const auto &relation : *operandRelations) {
            if (producer->getOperand(relation.operandNumber) != operand)
              continue;
            const auto *group = relation.groupForResultAxis(axis);
            if (!group || group->sourceAxes.size() > 1 ||
                (!group->sourceAxes.empty() && group->resultAxes.size() != 1))
              return failure();
            if (group->kind == FragmentAxisRelationKind::Reassociation &&
                group->sourceAxes.empty() &&
                !relation.isIntroducedUnitAxis(axis))
              return failure();
            std::optional<unsigned> selected = group->sourceAxes.empty()
                ? std::nullopt : std::optional<unsigned>(group->sourceAxes.front());
            if (found && (selected != operandAxis ||
                          introduced != group->sourceAxes.empty()))
              return failure();
            operandAxis = selected;
            introduced = group->sourceAxes.empty();
            found = true;
          }
          if (!found)
            return failure();
          if (introduced)
            return operand;
        } else if (reduction && !reductionSource) {
          BroadcastProjection relation = queryAxisProjection(input, fragment);
          if (!relation.isExact())
            return failure();
          operandAxis = relation.targetToSource[axis];
          if (!operandAxis)
            return operand;
        } else if (reduction) {
          SmallVector<unsigned> freeAxes;
          for (unsigned inputAxis = 0; inputAxis < input.getShape().size();
               ++inputAxis)
            if (!llvm::is_contained(reduction.getAxes(),
                                   static_cast<int64_t>(inputAxis)))
              freeAxes.push_back(inputAxis);
          if (axis < freeAxes.size())
            operandAxis = freeAxes[axis];
        }
        if (!operandAxis)
          return failure();
      }
      if (Value mapped = mapping.lookupOrNull(operand);
          mapped && !hasMultipleReplayAxes(operand.getType()))
        return mapped;
      return materialize(operand, operandAxis);
    };
    auto combineTail = [&](FragmentType target,
                           Value valid) -> FailureOr<Value> {
      if (!options.segmentTail)
        return valid ? FailureOr<Value>(valid)
                     : FailureOr<Value>(Value());
      FailureOr<Value> tail = projectPredicateToFragmentAxis(
          builder, location, options.segmentTail, target, axis);
      if (failed(tail))
        return failure();
      if (!valid)
        return *tail;
      return materializeValidityConjunction(builder, location, valid, *tail,
                                            target);
    };
    auto replayFill = [&](Value fill,
                          FragmentType target) -> FailureOr<Value> {
      if (fill) {
        FailureOr<Value> replayed = replayOperand(fill);
        if (failed(replayed))
          return failure();
        fill = *replayed;
        if (fill.getType() != target) {
          FailureOr<Value> projected =
              projectPhysicalValueToSchema(builder, location, fill, target);
          if (failed(projected))
            return failure();
          fill = *projected;
        }
        return fill;
      }
      if (!options.materializeZeroFill)
        return Value();
      return materializeZeroFragment(builder, location, target);
    };

    if (auto broadcast = dyn_cast<BroadcastOp>(producer)) {
      auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
      if (input && succeeded(operandRelations) && operandRelations->size() == 1) {
        const auto *group = operandRelations->front().groupForResultAxis(axis);
        if (group && group->sourceAxes.size() == 1) {
          unsigned inputAxis = group->sourceAxes.front();
          auto inputMap = cast<AxisMapAttr>(input.getAxisMaps()[inputAxis]);
          auto resultMap = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
          auto inputExtent =
              cast<PhysicalExprAttr>(input.getShape()[inputAxis]);
          bool sameLogicalAxis = inputMap.getDimensionId() > 0 &&
                                 inputMap.getDimensionId() ==
                                     resultMap.getDimensionId();
          bool nonUnitStaticExtent =
              inputExtent.getKind() ==
                  PhysicalExprKind::Constant &&
              inputExtent.getValue() > 1 &&
              operandRelations->front().hasCompatibleExtents();
          if (!(sourceAxisIdentity(inputMap) == sourceAxisIdentity(resultMap)) &&
              (sameLogicalAxis || nonUnitStaticExtent) &&
              input.getShape()[inputAxis] == fragment.getShape()[axis]) {
            // An extent-preserving projection can rename an occurrence. Replay
            // its input with that input's identity, retaining the output map.
            ReplayMaterializationOptions inputOptions = options;
            inputOptions.fragmentAxis = inputAxis;
            FailureOr<Value> replayed = materializeReplayedValue(
                builder, location, broadcast.getValue(),
                sourceAxisIdentity(inputMap), blockedExtent, mapping, insertionAnchor,
                inputOptions);
            if (failed(replayed))
              return failure();
            FailureOr<Value> projected = projectPhysicalValueToSchema(
                builder, location, *replayed, replaceReplayAxis(fragment, axis));
            if (failed(projected))
              return failure();
            remember(current, *projected, projection);
            return *projected;
          }
        }
      }
    }

    if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
      auto input = cast<FragmentType>(reshape.getValue().getType());
      if (auto inputAxis = reshapeInputAxis(reshape, axis)) {
        auto inputMap = cast<AxisMapAttr>(input.getAxisMaps()[*inputAxis]);
        auto resultMap = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
        if (!(sourceAxisIdentity(inputMap) == sourceAxisIdentity(resultMap)) &&
            input.getShape()[*inputAxis] == fragment.getShape()[axis]) {
          ReplayMaterializationOptions inputOptions = options;
          inputOptions.fragmentAxis = *inputAxis;
          FailureOr<Value> replayed = materializeReplayedValue(
              builder, location, reshape.getValue(), sourceAxisIdentity(inputMap),
              blockedExtent, mapping, insertionAnchor, inputOptions);
          if (failed(replayed))
            return failure();
          Value projected = builder.create<ReshapeOp>(
              location, replaceReplayAxis(fragment, axis), *replayed,
              reshape.getReassociation());
          remember(current, projected, projection);
          return projected;
        }
      }
    }

    if (auto contract = dyn_cast<ContractOp>(producer)) {
      auto axes = queryContractionAxes(contract);
      if (!axes || axis >= axes->results.size())
        return failure();
      // One SSA value can have different matrix roles at the two input uses.
      SmallVector<Value> operands{contract.getLhs(), contract.getRhs(),
                                   contract.getAccumulator()};
      auto replayAxis = [&](unsigned operand,
                            unsigned inputAxis) -> LogicalResult {
        auto type = cast<FragmentType>(operands[operand].getType());
        auto axisMap = cast<AxisMapAttr>(type.getAxisMaps()[inputAxis]);
        FailureOr<Value> replayed = failure();
        if (llvm::is_contained(replaySources, sourceAxisIdentity(axisMap))) {
          replayed = materialize(operands[operand], inputAxis);
        } else {
          ReplayMaterializationOptions inputOptions = options;
          inputOptions.fragmentAxis = inputAxis;
          replayed = materializeReplayedValue(
              builder, location, operands[operand], sourceAxisIdentity(axisMap),
              blockedExtent, mapping, insertionAnchor, inputOptions);
        }
        if (failed(replayed))
          return failure();
        operands[operand] = *replayed;
        return success();
      };
      const auto &resultAxis = axes->results[axis];
      unsigned operand = resultAxis.operand == ContractionOperand::Lhs ? 0 : 1;
      unsigned inputAxis = resultAxis.axis;
      if (failed(replayAxis(operand, inputAxis)) || failed(replayAxis(2, axis)))
        return failure();
      if (operand == 0)
        for (const auto &batch : axes->batch)
          if (batch.lhs == inputAxis && failed(replayAxis(1, batch.rhs)))
            return failure();
      IRMapping cloneMapping(mapping);
      auto cloned = cloneWithSchema(builder, producer, operands, cloneMapping,
                                   TypeRange{replaceReplayAxis(fragment, axis)});
      if (failed(cloned)) return failure();
      remember(current, (*cloned)[0], projection);
      return (*cloned)[0];
    }

    if (isa<BroadcastOp, SplatOp>(producer)) {
      FailureOr<Value> replayed = replayOperand(producer->getOperand(0));
      if (failed(replayed))
        return failure();
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, location, *replayed, replaceReplayAxis(fragment, axis));
      if (failed(projected))
        return failure();
      remember(current, *projected, projection);
      return *projected;
    }
    auto replayRead = [&](auto access, Value resource) -> FailureOr<Value> {
      if (auto load = dyn_cast<LoadOp>(access.getOperation());
          load && !canReplayReadAt(load, insertionAnchor))
        return failure();
      SmallVector<Value> coordinates;
      for (Value coordinate : access.getCoordinates()) {
        auto replayed = replayOperand(coordinate);
        if (failed(replayed))
          return failure();
        coordinates.push_back(*replayed);
      }
      Value valid;
      if (access.getValid()) {
        auto replayed = replayOperand(access.getValid());
        if (failed(replayed))
          return failure();
        valid = *replayed;
      }
      FragmentType resultType = replaceReplayAxis(fragment, axis);
      auto combined = combineTail(resultType, valid);
      if (failed(combined))
        return failure();
      auto fill = replayFill(access.getFill(), resultType);
      if (failed(fill))
        return failure();
      auto clone = builder.create<decltype(access)>(
          location, resultType, resource, coordinates, *combined, *fill,
          access.getSourceAxes());
      if (Attribute origin = access->getAttr(originAttr))
        clone->setAttr(originAttr, origin);
      remember(current, clone.getResult(), projection);
      return clone.getResult();
    };
    if (auto load = dyn_cast<LoadOp>(producer))
      return replayRead(load, load.getResource());
    if (auto gather = dyn_cast<GatherOp>(producer)) {
      auto source = replayOperand(gather.getSource());
      if (failed(source))
        return failure();
      return replayRead(gather, *source);
    }
    bool pureBranch = isa<scf::IfOp>(producer) &&
                      options.scope == PhysicalReplayScope::ValueGraph &&
                      isMemoryEffectFree(producer);
    if (!pureBranch && !isPhysicalReplayNode(producer, options.scope,
                                            /*allowAccesses=*/false))
      return failure();
    IRMapping cloneMapping(mapping);
    llvm::SetVector<Value> replayOperands(producer->operand_begin(),
                                         producer->operand_end());
    if (pureBranch)
      for (Region &region : producer->getRegions())
        getUsedValuesDefinedAbove(region, replayOperands);
    for (Value operand : replayOperands) {
      FailureOr<Value> replayed = replayOperand(operand);
      if (failed(replayed))
        return failure();
      cloneMapping.map(operand, *replayed);
    }
    FragmentType pointwiseType;
    if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer)) {
      SmallVector<Value> operands;
      for (Value operand : producer->getOperands())
        operands.push_back(cloneMapping.lookupOrDefault(operand));
      auto refined = queryValueSchema(
          kernel, replaceReplayAxis(fragment, axis), operands);
      if (failed(refined))
        return failure();
      pointwiseType = *refined;
    }
    bool structuredResults = pureBranch || isa<ReduceOp, ScanOp>(producer);
    bool introducedUnitAxis = isIntroducedReshapeUnitAxis(current, axis);
    auto cloned = cloneWithPhysicalSchema(builder, producer, cloneMapping, [&](Value original) -> Type {
          bool directResult = original.getDefiningOp() == producer;
          Type type = original.getType();
          if (!directResult)
            return structuredResults
                ? replayHelperType(original, cast<PhysicalExprAttr>(fragment.getShape()[axis]))
                : type;
          if (structuredResults)
            type = replaceReplayType(type);
          if (original != current)
            return type;
          if (pointwiseType)
            return pointwiseType;
          auto selected = dyn_cast<FragmentType>(type);
          if (!selected || axis >= selected.getShape().size())
            return {};
          return introducedUnitAxis ? type : replaceReplayAxis(selected, axis);
        });
    if (failed(cloned)) return failure();
    for (auto [original, resultValue] :
         llvm::zip(producer->getResults(), *cloned)) {
      if (!hasMultipleReplayAxes(original.getType()))
        mapping.map(original, resultValue);
    }
    auto result = dyn_cast<OpResult>(current);
    if (!result || result.getResultNumber() >= cloned->size())
      return failure();
    Value clonedValue = (*cloned)[result.getResultNumber()];
    auto clonedType = dyn_cast<FragmentType>(clonedValue.getType());
    if (!clonedType || axis >= clonedType.getShape().size())
      return failure();
    remember(current, clonedValue, projection);
    return clonedValue;
  };

  FailureOr<Value> result = materialize(value, options.fragmentAxis);
  if (projectionFailed || failed(result)) return failure();
  resultMapping = std::move(mapping);
  insertion.commit();
  return result;
}

} // namespace intent::gpu
