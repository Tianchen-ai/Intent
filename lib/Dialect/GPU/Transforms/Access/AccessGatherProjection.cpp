#include "AccessComposition.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::gpu::access {

namespace {

bool scalarPredicateContains(Value predicate, Value required) {
  if (!predicate)
    return false;
  if (predicate == required)
    return true;
  auto lhs = predicate.getDefiningOp<CompareOp>();
  auto rhs = required.getDefiningOp<CompareOp>();
  if (lhs && rhs && lhs.getPredicate() == rhs.getPredicate() &&
      lhs.getLhs().getType().isIndex() && rhs.getLhs().getType().isIndex() &&
      samePhysicalScalarExpression(lhs.getLhs(), rhs.getLhs()) &&
      samePhysicalScalarExpression(lhs.getRhs(), rhs.getRhs()))
    return true;
  auto binary = predicate.getDefiningOp<BinaryOp>();
  return binary && binary.getOperatorKind() == BinaryOperator::LogicalAnd &&
         (scalarPredicateContains(binary.getLhs(), required) ||
          scalarPredicateContains(binary.getRhs(), required));
}

bool impliesUniformPredicate(Value predicate, Value required) {
  if (!required)
    return true;
  auto constant = dyn_cast_or_null<IntegerAttr>(
      UniformValueAnalysis(describeUniformValue).evaluate(required));
  if (constant && constant.getType().isInteger(1) &&
      constant.getValue().isOne())
    return true;
  if (auto splat = required.getDefiningOp<SplatOp>())
    return impliesUniformPredicate(predicate, splat.getValue());
  if (auto broadcast = required.getDefiningOp<BroadcastOp>())
    return impliesUniformPredicate(predicate, broadcast.getValue());
  if (auto binary = required.getDefiningOp<BinaryOp>();
      binary && binary.getOperatorKind() == BinaryOperator::LogicalAnd)
    return impliesUniformPredicate(predicate, binary.getLhs()) &&
           impliesUniformPredicate(predicate, binary.getRhs());
  // Only scalar conjuncts may be shared across retained fragment lanes.
  return required.getType().isInteger(1) &&
         scalarPredicateContains(predicate, required);
}

} // namespace

bool reuseFragmentGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  if (!source || gather.getResult().getType() != source.getElementType() ||
      gather.getSourceAxes().size() != source.getShape().size() ||
      llvm::any_of(gather.getCoordinates(),
                   [](Value coordinate) { return !coordinate.getType().isIndex(); }))
    return false;
  for (auto [axis, sourceAxis] : llvm::enumerate(gather.getSourceAxes()))
    if (sourceAxis != static_cast<int64_t>(axis))
      return false;

  // A scalar extraction can reuse an earlier immutable slice of the same SSA
  // tensor. Native layout/communication still belongs to the provider compiler.
  for (Operation *user : gather.getSource().getUsers()) {
    auto available = dyn_cast<GatherOp>(user);
    auto sliced = available
                      ? dyn_cast<FragmentType>(available.getResult().getType())
                      : FragmentType();
    if (!available || available == gather ||
        available->getBlock() != gather->getBlock() ||
        !available->isBeforeInBlock(gather) || !sliced ||
        sliced.getOwner() != source.getOwner() ||
        available.getSourceAxes().empty() ||
        sliced.getShape().size() + available.getSourceAxes().size() !=
            source.getShape().size() ||
        !impliesUniformPredicate(gather.getValid(), available.getValid()))
      continue;
    SmallVector<bool> selected(source.getShape().size(), false);
    bool matches = true;
    for (auto [coordinate, axis] :
         llvm::zip(available.getCoordinates(), available.getSourceAxes())) {
      if (axis < 0 || axis >= static_cast<int64_t>(selected.size()) ||
          selected[axis] || !coordinate.getType().isIndex() ||
          !samePhysicalScalarExpression(coordinate,
                                        gather.getCoordinates()[axis])) {
        matches = false;
        break;
      }
      selected[axis] = true;
    }
    if (!matches)
      continue;
    SmallVector<Value> coordinates;
    SmallVector<int64_t> axes;
    for (unsigned axis = 0; axis < selected.size(); ++axis) {
      if (selected[axis])
        continue;
      unsigned retained = coordinates.size();
      auto originalMap = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
      auto retainedMap = cast<AxisMapAttr>(sliced.getAxisMaps()[retained]);
      if (sliced.getShape()[retained] != source.getShape()[axis] ||
          !(sourceAxisIdentity(originalMap) == sourceAxisIdentity(retainedMap)) ||
          originalMap.getDimensionId() != retainedMap.getDimensionId()) {
        matches = false;
        break;
      }
      coordinates.push_back(gather.getCoordinates()[axis]);
      axes.push_back(retained);
    }
    if (!matches)
      continue;
    OpBuilder builder(gather);
    auto replacement = builder.create<GatherOp>(
        gather.getLoc(), gather.getResult().getType(), available.getResult(),
        coordinates, gather.getValid(), gather.getFill(), axes);
    replacement->setDiscardableAttrs(
        llvm::to_vector(gather->getDiscardableAttrs()));
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
    return true;
  }
  return false;
}

FailureOr<bool> composeIdentityFragmentGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!source || !result || source.getOwner() != result.getOwner() ||
      gather.getCoordinates().size() != source.getShape().size() ||
      gather.getSourceAxes().size() != source.getShape().size())
    return false;
  // Gather indexes fragment positions. A complete one-dimensional ordinal
  // range is an identity even when storage and consumer use different axes.
  if (source.getShape().size() == 1 && source.getShape() == result.getShape() &&
      gather.getSourceAxes() == ArrayRef<int64_t>{0}) {
    auto range = gather.getCoordinates().front().getDefiningOp<MakeRangeOp>();
    if (range && range.getResult().getType().getShape() == result.getShape() &&
        range.getResult().getType().getAxisMaps() == result.getAxisMaps() &&
        isZero(range.getStart()) &&
        isUnitStepRange(range) &&
        queryLaunchExpression(range.getExtent()) == source.getShape()[0]) {
      OpBuilder builder(gather);
      auto group = ReshapeGroupAttr::get(
          builder.getContext(), builder.getDenseI64ArrayAttr({0}),
          builder.getDenseI64ArrayAttr({0}));
      Value replacement = builder.create<ReshapeOp>(
          gather.getLoc(), result, gather.getSource(), builder.getArrayAttr({group}));
      if (gather.getValid())
        replacement = builder.create<SelectOp>(gather.getLoc(), result,
            gather.getValid(), replacement, gather.getFill());
      if (Attribute origin = gather->getAttr(originAttr))
        replacement.getDefiningOp()->setAttr(originAttr, origin);
      gather.getResult().replaceAllUsesWith(replacement);
      gather.erase();
      return true;
    }
  }
  SmallVector<bool> represented(result.getShape().size(), false);
  for (auto [coordinate, sourceAxis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(source.getShape().size()))
      return false;
    auto expected = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    PhysicalSourceAxis physicalSource = sourceAxisIdentity(expected);
    PhysicalAxisProjection resultAxis =
        queryFragmentAxis(result, physicalSource, expected.getDimensionId());
    // Coordinate broadcasting does not change the ordinal of a full slice.
    // Inspect the range before its singleton axes were inserted.
    while (auto broadcast = coordinate.getDefiningOp<BroadcastOp>())
      coordinate = broadcast.getValue();
    PhysicalAxisProjection coordinateAxis =
        queryCoordinateIndex(ValueRange{coordinate}, physicalSource,
                             expected.getDimensionId());
    auto resultMapping =
        resultAxis.isExact()
            ? dyn_cast<AxisMapAttr>(result.getAxisMaps()[resultAxis.fragmentAxis])
            : AxisMapAttr();
    auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
    auto coordinateRange = coordinate.getDefiningOp<MakeRangeOp>();
    if (!resultAxis.isExact() || !coordinateAxis.isExact() || !resultMapping ||
        resultMapping.getDimensionId() != expected.getDimensionId() ||
        coordinateAxis.dimensionId != expected.getDimensionId() ||
        !coordinateType || !coordinateRange ||
        !isZero(coordinateRange.getStart()) ||
        !isUnitStepRange(coordinateRange) ||
        source.getShape()[sourceAxis] !=
            result.getShape()[resultAxis.fragmentAxis] ||
        coordinateType.getShape().size() != 1 ||
        coordinateType.getShape()[0] != source.getShape()[sourceAxis] ||
        coordinateAxis.fragmentAxis != 0)
      return false;
    represented[resultAxis.fragmentAxis] = true;
  }
  for (auto [axis, extent] : llvm::enumerate(result.getShape())) {
    if (represented[axis])
      continue;
    auto expression = dyn_cast<PhysicalExprAttr>(extent);
    if (!expression ||
        expression.getKind() !=
            PhysicalExprKind::Constant ||
        expression.getValue() != 1)
      return false;
  }
  OpBuilder builder(gather);
  Value replacement = builder.create<BroadcastOp>(gather.getLoc(), result,
                                                   gather.getSource());
  if (gather.getValid())
    replacement = builder.create<SelectOp>(gather.getLoc(), result,
                                           gather.getValid(), replacement,
                                           gather.getFill());
  if (Attribute origin = gather->getAttr(originAttr))
    replacement.getDefiningOp()->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement);
  gather.erase();
  return true;
}

FailureOr<bool> projectFragmentGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!source || !result ||
      gather.getCoordinates().size() != source.getShape().size() ||
      gather.getSourceAxes().size() != source.getShape().size())
    return false;

  SmallVector<Value> selectedCoordinates;
  SmallVector<int64_t> selectedAxes;
  bool projected = false;
  for (auto [coordinate, sourceAxis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(source.getShape().size()))
      return false;
    auto expected = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    PhysicalSourceAxis physicalSource = sourceAxisIdentity(expected);
    PhysicalAxisProjection resultAxis =
        queryFragmentAxis(result, physicalSource, expected.getDimensionId());
    if (resultAxis.state == PhysicalFactState::Ambiguous)
      return false;
    if (!resultAxis.isExact()) {
      selectedCoordinates.push_back(coordinate);
      selectedAxes.push_back(sourceAxis);
      continue;
    }
    PhysicalAxisProjection coordinateAxis =
        queryCoordinateIndex(ValueRange{coordinate}, physicalSource,
                             expected.getDimensionId());
    auto resultMapping =
        dyn_cast<AxisMapAttr>(result.getAxisMaps()[resultAxis.fragmentAxis]);
    auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
    if (!coordinateAxis.isExact() || !resultMapping ||
        resultMapping.getDimensionId() != expected.getDimensionId() ||
        coordinateAxis.dimensionId != expected.getDimensionId() ||
        !coordinateType ||
        coordinateType.getShape().size() != 1 ||
        source.getShape()[sourceAxis] !=
            result.getShape()[resultAxis.fragmentAxis] ||
        coordinateType.getShape()[0] != source.getShape()[sourceAxis] ||
        coordinateAxis.fragmentAxis != 0)
      return false;
    projected = true;
  }
  if (!projected || selectedCoordinates.empty())
    return false;

  OpBuilder builder(gather);
  auto replacement = builder.create<GatherOp>(
      gather.getLoc(), result, gather.getSource(), selectedCoordinates,
      gather.getValid(), gather.getFill(), selectedAxes);
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  return true;
}

FailureOr<bool> composeBroadcastGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!source || !result || !gather.getValid() ||
      source.getOwner() != result.getOwner() ||
      gather.getCoordinates().size() != source.getShape().size())
    return false;

  SmallVector<Value> coordinates;
  FragmentType indexSchema;
  for (Value coordinate : gather.getCoordinates()) {
    if (!uniformElementType(coordinate.getType()).isIndex())
      return false;
    Value scalar = coordinate;
    while (isa<FragmentType>(scalar.getType())) {
      UniformExpression expression = describeUniformValue(scalar);
      if (expression.kind != UniformKind::Forward || expression.operands.size() != 1)
        break;
      scalar = expression.operands.front();
    }
    if (!isa<FragmentType>(scalar.getType()))
      coordinate = scalar;
    if (auto fragment = dyn_cast<FragmentType>(coordinate.getType());
        fragment && (!indexSchema || fragment.getShape().size() >
                                       indexSchema.getShape().size()))
      indexSchema = fragment;
    coordinates.push_back(coordinate);
  }
  if (indexSchema) {
    if (indexSchema.getOwner() != result.getOwner() ||
        indexSchema.getShape().size() >= result.getShape().size())
      return false;
    auto expansion = queryBroadcastProjection(indexSchema, result);
    if (!expansion.isExact())
      return false;
    for (Value coordinate : coordinates) {
      auto fragment = dyn_cast<FragmentType>(coordinate.getType());
      if (!fragment)
        continue;
      auto projection = queryBroadcastProjection(fragment, indexSchema);
      auto original = queryBroadcastProjection(fragment, result);
      if (!projection.isExact() || !original.isExact())
        return false;
      for (auto [axis, indexAxis] : llvm::enumerate(expansion.targetToSource))
        if ((indexAxis ? projection.targetToSource[*indexAxis] : std::nullopt) !=
            original.targetToSource[axis])
          return false;
    }
  }

  OpBuilder builder(gather);
  Location location = gather.getLoc();
  auto selectedTypeFor = [&](Type element) -> Type {
    return indexSchema ? Type(FragmentType::get(
        result.getContext(), element, indexSchema.getShape(),
        indexSchema.getAxisMaps(), indexSchema.getValidity(), result.getOwner()))
        : element;
  };
  Type selectedType = selectedTypeFor(result.getElementType());
  Type indexType = selectedTypeFor(builder.getIndexType());
  auto valid = gatherBounds(builder, location, source, coordinates,
                            gather.getSourceAxes(), indexType);
  if (failed(valid))
    return gather.emitOpError("gather lost its exact index broadcast relation");
  Value fill;
  if (auto fragment = dyn_cast<FragmentType>(selectedType)) {
    auto zeroFill = materializeZeroFragment(builder, location, fragment);
    if (failed(zeroFill))
      return failure();
    fill = *zeroFill;
  } else {
    fill = builder.create<arith::ConstantOp>(location, builder.getZeroAttr(selectedType));
  }
  // Only indices determine the immutable SSA read. Keep its own bounds before
  // broadcasting; the original lane-dependent predicate and fill remain below.
  Value selected = builder.create<GatherOp>(
      location, selectedType, gather.getSource(), coordinates, *valid, fill,
      gather.getSourceAxes());
  Value expanded = builder.create<BroadcastOp>(location, result, selected);
  auto replacement = builder.create<SelectOp>(
      location, result, gather.getValid(), expanded, gather.getFill());
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  return true;
}

} // namespace intent::gpu::access
