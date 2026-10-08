#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Worklist.h"

using namespace mlir;

namespace intent::gpu::value_relations {

WalkResult alignAccessResult(Operation *operation, RelationWorklist &changes) {
  auto kernel = operation->getParentOfType<func::FuncOp>();
  auto access = cast<AccessOpInterface>(operation);
  ValueRange coordinates = access.getAccessCoordinates();
  Value result = access.getAccessResult();
  auto current = dyn_cast<FragmentType>(result.getType());
  if (!current)
    return WalkResult::advance();
  FailureOr<FragmentType> refined = queryAccessResultSchema(kernel, access);
  if (failed(refined)) {
    InFlightDiagnostic diagnostic = operation->emitOpError(
        "access result has no unique physical coordinate projection");
    diagnostic << "; result=" << current;
    for (Value coordinate : coordinates)
      diagnostic << "; coordinate=" << coordinate.getType();
    return WalkResult::interrupt();
  }
  for (unsigned axis = 0; axis < (*refined).getShape().size(); ++axis) {
    if (axis >= current.getShape().size() ||
        current.getShape()[axis] == (*refined).getShape()[axis])
      continue;
    if (failed(retargetFragmentAxisExtent(
            result, axis, cast<PhysicalExprAttr>((*refined).getShape()[axis]),
            changes.typeChanged(), &changes)))
      return WalkResult::interrupt();
  }
  changes.setType(result, *refined);
  return WalkResult::advance();
}

LogicalResult alignAccessValue(Operation *operation,
                               RelationWorklist &changes) {
  auto kernel = operation->getParentOfType<func::FuncOp>();
  auto predicateType = [&](FragmentType value) {
    return FragmentType::get(kernel.getContext(),
                             IntegerType::get(kernel.getContext(), 1),
                             value.getShape(), value.getAxisMaps(),
                             value.getValidity(), value.getOwner());
  };
  auto project = [&](OpBuilder &builder, Location location, Value value,
                     Type target) -> FailureOr<Value> {
    if (!value)
      return failure();
    return projectPhysicalValueToSchema(builder, location, value, target,
                                        changes.typeChanged());
  };
  auto alignCoordinates = [&](AccessOpInterface access,
                              Type valueType) -> LogicalResult {
    auto payload = dyn_cast<FragmentType>(valueType);
    if (!payload)
      return success();
    for (auto [slot, coordinate] :
         llvm::enumerate(access.getAccessCoordinates())) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type)
        continue;
      PhysicalProgramAnalysis analysis(kernel);
      SmallVector<Attribute> shape(type.getShape().begin(),
                                   type.getShape().end());
      bool changed = false;
      auto relation = queryAccessCoordinateAxes(access, slot);
      if (relation.state == BroadcastProjectionState::Ambiguous)
        return access.emitOpError("coordinate has ambiguous payload axes");
      for (auto [payloadAxis, coordinateAxis] :
           llvm::enumerate(relation.targetToSource)) {
        if (!coordinateAxis)
          continue;
        unsigned axis = *coordinateAxis;
        auto mapping = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
        auto payloadMapping =
            cast<AxisMapAttr>(payload.getAxisMaps()[payloadAxis]);
        // The access owns the axis occurrence. Extent equality or a repeated
        // source identity alone cannot select a coordinate's payload slot.
        if (mapping.getDimensionId() != payloadMapping.getDimensionId() ||
            shape[axis] == payload.getShape()[payloadAxis])
          continue;
        PhysicalRangeFact ranges = analysis.axisRanges(coordinate, axis);
        if (!ranges.isExact() || !ranges.roots.empty())
          continue;
        shape[axis] = payload.getShape()[payloadAxis];
        changed = true;
      }
      auto permutation = queryAxisPermutation(type, payload);
      bool permuted =
          permutation &&
          llvm::any_of(llvm::enumerate(*permutation), [](auto item) {
            return item.value() != static_cast<int64_t>(item.index());
          });
      if (!changed && !permuted)
        continue;
      auto target = FragmentType::get(
          kernel.getContext(), type.getElementType(),
          permuted ? payload.getShape()
                   : ArrayAttr::get(kernel.getContext(), shape),
          permuted ? payload.getAxisMaps() : type.getAxisMaps(),
          type.getValidity(), type.getOwner());
      OpBuilder builder(access);
      builder.setListener(&changes);
      FailureOr<Value> aligned =
          project(builder, access.getLoc(), coordinate, target);
      if (failed(aligned))
        return access.emitOpError(
                   "cannot align coordinate with its access schema")
               << "; coordinate=" << coordinate;
      access.getAccessCoordinatesMutable().slice(slot, 1).assign(*aligned);
    }
    return success();
  };

  auto access = cast<AccessOpInterface>(operation);
  auto alignRead = [&]() -> LogicalResult {
    if (failed(alignCoordinates(access, access.getAccessValueType())))
      return failure();
    if (!access.getAccessValidity() && !access.getAccessFill())
      return success();
    if (!access.getAccessValidity())
      return access.emitOpError("fill has no validity authority");
    OpBuilder builder(access);
    builder.setListener(&changes);
    Type valueType = access.getAccessValueType();
    auto fragment = dyn_cast<FragmentType>(valueType);
    if (access.getAccessKind() == AccessKind::Gather && !fragment) {
      if (access.getAccessFill())
        return success();
      return access.emitOpError(
          "scalar gather validity and fill must remain paired");
    }
    Type validType =
        fragment ? Type(predicateType(fragment)) : builder.getI1Type();
    auto valid = project(builder, access.getLoc(), access.getAccessValidity(),
                         validType);
    if (failed(valid))
      return access.emitOpError(
          "cannot align read validity with its value schema");
    auto fill = access.getAccessFill()
                    ? project(builder, access.getLoc(), access.getAccessFill(),
                              valueType)
                    : materializeZeroValue(builder, access.getLoc(), valueType);
    if (failed(fill))
      return access.emitOpError("cannot align read fill with its value schema");
    // Keep the access identity, coordinates, effects and all attributes intact;
    // only its already-established validity/fill relation changes here.
    return access.updateAccessOperands(access.getAccessCoordinates(),
                                       access.getAccessPayloads(), *valid,
                                       *fill);
  };
  if (access.getAccessKind() == AccessKind::Load ||
      access.getAccessKind() == AccessKind::Gather)
    return alignRead();
  auto store = dyn_cast<StoreOp>(operation);
  if (!store)
    return success();
  if (failed(alignCoordinates(access, store.getValue().getType())))
    return failure();
  if (!store.getValid())
    return success();
  auto currentType = dyn_cast<FragmentType>(store.getValue().getType());
  if (!currentType) {
    auto validity = dyn_cast<FragmentType>(store.getValid().getType());
    if (!validity || !llvm::all_of(validity.getShape(), [](Attribute extent) {
          return constantPhysicalExpression(cast<PhysicalExprAttr>(extent)) ==
                 1;
        }))
      return success();
    // Ownership can retain a one-lane predicate for a scalar write.  Adopt
    // that schema without widening the write into additional lanes.
    currentType = FragmentType::get(
        kernel.getContext(), store.getValue().getType(), validity.getShape(),
        validity.getAxisMaps(), validity.getValidity(), validity.getOwner());
    OpBuilder builder(store);
    builder.setListener(&changes);
    FailureOr<Value> value =
        project(builder, store.getLoc(), store.getValue(), currentType);
    if (failed(value))
      return store.emitOpError(
          "cannot align scalar store with its one-lane validity");
    store.getValueMutable().assign(*value);
  }
  // A value whose extent is already selected by an exact range or verified
  // reshape is the physical data authority for the write.  Retarget the
  // address relation to that extent before reconciling schemas.  This keeps
  // extents and coordinate provenance separate: the value does not acquire
  // the view's source identity, and the address does not overwrite a verified
  // row-major reshape decision.
  for (unsigned axis = 0; axis < currentType.getShape().size(); ++axis) {
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalAxisRealizationFact realization =
        analysis.axisRealization(store.getValue(), axis);
    if (!realization.hasExtentAuthority())
      continue;
    auto mapping = cast<AxisMapAttr>(currentType.getAxisMaps()[axis]);
    auto extent = cast<PhysicalExprAttr>(currentType.getShape()[axis]);
    for (auto [slot, coordinate] : llvm::enumerate(store.getCoordinates())) {
      auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
      if (!coordinateType)
        continue;
      auto relation = queryAccessCoordinateAxes(access, slot);
      if (relation.state == BroadcastProjectionState::Ambiguous)
        return store.emitOpError("store coordinate has ambiguous payload axes");
      if (axis >= relation.targetToSource.size() ||
          !relation.targetToSource[axis])
        continue;
      unsigned coordinateAxis = *relation.targetToSource[axis];
      auto coordinateMapping =
          cast<AxisMapAttr>(coordinateType.getAxisMaps()[coordinateAxis]);
      if (coordinateMapping.getDimensionId() != mapping.getDimensionId() ||
          coordinateType.getShape()[coordinateAxis] == extent)
        continue;
      if (failed(retargetFragmentAxisExtent(coordinate, coordinateAxis, extent,
                                            changes.typeChanged(), &changes)))
        return failure();
    }
  }
  currentType = cast<FragmentType>(store.getValue().getType());
  OpBuilder builder(store);
  builder.setListener(&changes);
  FailureOr<FragmentType> valueType = queryAccessResultSchema(
      kernel, cast<AccessOpInterface>(store.getOperation()));
  if (failed(valueType))
    return store.emitOpError(
        "store value has no unique physical coordinate projection");
  for (auto [axis, mapping] : llvm::enumerate((*valueType).getAxisMaps())) {
    if (axis >= currentType.getShape().size() ||
        currentType.getShape()[axis] == (*valueType).getShape()[axis])
      continue;
    int64_t dimension = cast<AxisMapAttr>(mapping).getDimensionId();
    if (dimension <= 0)
      return store.emitOpError(
          "store coordinate refinement has no logical dimension authority");
    if (failed(retargetFragmentAxisExtent(
            store.getValue(), axis,
            cast<PhysicalExprAttr>((*valueType).getShape()[axis]),
            changes.typeChanged(), &changes)))
      return failure();
  }
  FailureOr<Value> value =
      project(builder, store.getLoc(), store.getValue(), *valueType);
  if (failed(value)) {
    InFlightDiagnostic diagnostic = store.emitOpError(
        "cannot align store value with its coordinate schema");
    diagnostic << "; value=" << store.getValue().getType()
               << "; coordinate_schema=" << *valueType;
    for (Value coordinate : store.getCoordinates())
      diagnostic << "; coordinate=" << coordinate.getType();
    return failure();
  }
  FailureOr<Value> valid = project(builder, store.getLoc(), store.getValid(),
                                   predicateType(*valueType));
  if (failed(valid)) {
    InFlightDiagnostic diagnostic =
        store.emitOpError("cannot align store validity with its value schema");
    diagnostic << "; value=" << *valueType
               << "; validity=" << store.getValid().getType();
    if (Operation *producer = store.getValue().getDefiningOp())
      diagnostic << "; value_producer=" << producer->getName();
    return failure();
  }
  // Preserve the access identity and every effect/attribute while closing
  // only its payload and predicate schemas.
  store.getValueMutable().assign(*value);
  store.getValidMutable().assign(*valid);
  return success();
}

} // namespace intent::gpu::value_relations
