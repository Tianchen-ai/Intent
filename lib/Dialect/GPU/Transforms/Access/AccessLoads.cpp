#include "AccessComposition.h"
#include "../Value/ScopePlacement.h"
#include "Intent/Dialect/GPU/Analysis/ConfigurationExpressions.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::gpu::access {

FailureOr<bool> composeSelectLoad(SelectOp select) {
  auto resultType = dyn_cast<FragmentType>(select.getResult().getType());
  Value loaded = select.getTrueValue();
  SmallVector<Operation *> projections;
  while (Operation *operation = loaded.getDefiningOp()) {
    if (!isa<BroadcastOp, TransposeOp, ReshapeOp>(operation))
      break;
    auto source = dyn_cast<FragmentType>(operation->getOperand(0).getType());
    auto result = dyn_cast<FragmentType>(loaded.getType());
    if (!source || !result || !loaded.hasOneUse())
      return false;
    auto relations = queryFragmentOperandRelations(operation);
    if (failed(relations))
      return false;
    if (isa<TransposeOp, ReshapeOp>(operation)) {
      for (const FragmentAxisGroup &group : relations->front().groups) {
        ArrayRef<unsigned> from = group.sourceAxes;
        ArrayRef<unsigned> to = group.resultAxes;
        if (from.empty() || to.empty()) {
          if (!unitAxes(source, from) || !unitAxes(result, to))
            return false;
          continue;
        }
        if (from.size() != 1 || to.size() != 1)
          return false;
        unsigned sourceAxis = from.front();
        unsigned resultAxis = to.front();
        auto original = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
        auto target = cast<AxisMapAttr>(result.getAxisMaps()[resultAxis]);
        if (!(sourceAxisIdentity(original) == sourceAxisIdentity(target)) ||
            original.getDimensionId() != target.getDimensionId() ||
            source.getShape()[sourceAxis] != result.getShape()[resultAxis])
          return false;
      }
    } else {
      if (source.getShape() != result.getShape() ||
          llvm::any_of(relations->front().groups, [](const FragmentAxisGroup &group) {
            return group.sourceAxes.size() != 1 || group.resultAxes.size() != 1 ||
                   group.sourceAxes.front() != group.resultAxes.front();
          }))
        return false;
    }
    projections.push_back(operation);
    loaded = operation->getOperand(0);
  }
  auto load = loaded.getDefiningOp<LoadOp>();
  if (!resultType || !load || !load.getResult().hasOneUse() ||
      !canReplayReadAt(load, select))
    return false;

  OpBuilder builder(select);
  auto projectBack = [&](Value value) -> FailureOr<Value> {
    for (Operation *projection : projections) {
      auto source = cast<FragmentType>(projection->getOperand(0).getType());
      auto result = cast<FragmentType>(projection->getResult(0).getType());
      Type element = uniformElementType(value.getType());
      auto sourceSchema = FragmentType::get(
          source.getContext(), element, source.getShape(), source.getAxisMaps(),
          source.getValidity(), source.getOwner());
      if (auto transpose = dyn_cast<TransposeOp>(projection)) {
        auto resultSchema = FragmentType::get(
            result.getContext(), element, result.getShape(), result.getAxisMaps(),
            result.getValidity(), result.getOwner());
        auto projected = projectPhysicalValueToSchema(
            builder, select.getLoc(), value, resultSchema);
        if (failed(projected))
          return failure();
        SmallVector<int64_t> inverse(transpose.getPermutation().size());
        for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation()))
          inverse[sourceAxis] = axis;
        value = builder.create<TransposeOp>(select.getLoc(), sourceSchema,
                                            *projected, inverse);
      } else if (auto reshape = dyn_cast<ReshapeOp>(projection)) {
        auto resultSchema = FragmentType::get(
            result.getContext(), element, result.getShape(), result.getAxisMaps(),
            result.getValidity(), result.getOwner());
        auto projected = projectPhysicalValueToSchema(
            builder, select.getLoc(), value, resultSchema);
        if (failed(projected))
          return failure();
        SmallVector<Attribute> inverse;
        for (Attribute attribute : reshape.getReassociation()) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          inverse.push_back(ReshapeGroupAttr::get(
              select.getContext(), group.getResultAxes(), group.getSourceAxes()));
        }
        value = builder.create<ReshapeOp>(
            select.getLoc(), sourceSchema, *projected, builder.getArrayAttr(inverse));
      } else {
        auto projected = projectPhysicalValueToSchema(
            builder, select.getLoc(), value, sourceSchema);
        if (failed(projected))
          return failure();
        value = *projected;
      }
    }
    return value;
  };
  auto condition = projectBack(select.getCondition());
  auto projectedFill = projectBack(select.getFalseValue());
  if (failed(condition) || failed(projectedFill))
    return false;
  Value fill = *projectedFill;
  if (load.getValid()) {
    if (!load.getFill())
      return false;
  } else if (load.getFill()) {
    return false;
  }

  auto loadedType = projections.empty()
                        ? resultType
                        : cast<FragmentType>(load.getResult().getType());
  FailureOr<Value> valid = combinePredicates(
      builder, select.getLoc(), loadedType, load.getValid(), *condition);
  if (failed(valid)) {
    select.emitOpError(
        "masked load predicate cannot adopt the loaded value relation");
    return failure();
  }
  if (fill.getType() != loadedType) {
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, select.getLoc(), fill, loadedType);
    if (failed(projected)) {
      select.emitOpError(
          "masked load fill cannot adopt the loaded value relation");
      return failure();
    }
    fill = *projected;
  }
  if (load.getValid() && !sameUniformValue(load.getFill(), fill)) {
    auto sourceFill = projectPhysicalValueToSchema(
        builder, select.getLoc(), load.getFill(), loadedType);
    auto predicate = combinePredicates(
        builder, select.getLoc(), loadedType, Value(), *condition);
    if (failed(sourceFill) || failed(predicate))
      return false;
    fill = builder.create<SelectOp>(select.getLoc(), loadedType, *predicate,
                                    *sourceFill, fill);
  }
  auto replacement = builder.create<LoadOp>(
      select.getLoc(), loadedType, load.getResource(), load.getCoordinates(),
      *valid, fill, load.getSourceAxes());
  if (Attribute origin = load->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  // Keep the original load and value schemas. Only the predicate and fill
  // travel backwards through the inverse projection.
  IRMapping mapping;
  mapping.map(load.getResult(), replacement.getResult());
  for (Operation *projection : llvm::reverse(projections))
    builder.clone(*projection, mapping);
  select.getResult().replaceAllUsesWith(
      mapping.lookupOrDefault(select.getTrueValue()));
  select.erase();
  for (Operation *projection : projections)
    projection->erase();
  load.erase();
  return true;
}

namespace {

bool hasFullSnapshotConsumer(Value value, Operation *anchor) {
  auto kernel = anchor->getParentOfType<func::FuncOp>();
  DominanceInfo dominance(kernel);
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!visited.insert(current).second) continue;
    for (Operation *user : current.getUsers()) {
      if (user == anchor || anchor->isAncestor(user)) continue;
      if (auto gather = dyn_cast<GatherOp>(user);
          gather && gather.getSource() == current) continue;
      if (isa<FragmentOpInterface>(user) && !user->getNumRegions()) {
        // These values can disappear when all their consumers take sub-tiles.
        llvm::append_range(pending, user->getResults());
        continue;
      }
      if (llvm::any_of(user->getResultTypes(), [](Type type) {
            return isa<FragmentType>(type);
          }) && dominance.properlyDominates(user, anchor)) return true;
    }
  }
  return false;
}

bool isAlignedSnapshotSlice(GatherOp gather) {
  auto load = gather.getSource().getDefiningOp<LoadOp>();
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!load || !source || !result || source.getShape().empty() ||
      source.getShape().size() != result.getShape().size() ||
      source.getElementType() != result.getElementType() ||
      source.getValidity() != result.getValidity() || source.getOwner() != result.getOwner() ||
      gather.getCoordinates().size() != 1 || gather.getSourceAxes().size() != 1 ||
      gather.getSourceAxes().front() != static_cast<int64_t>(source.getShape().size() - 1) ||
      !hasFullSnapshotConsumer(load.getResult(), gather) || !canReplayReadAt(load, gather))
    return false;
  auto kernel = gather->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  for (unsigned position = 0; position < source.getShape().size(); ++position) {
    auto realization = analysis.axisRealization(load.getResult(), position);
    if (!realization.isExact() || !realization.physicalized ||
        realization.constructionScalarSeed) return false;
  }
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities) return false;
  auto footprint = minimumFragmentRegisterFootprint(
      kernel, load.getResult(), capabilities.getRegistersPerUnit(),
      FragmentFootprintScope::PhysicalShape);
  // As in replay retention, this is an opportunity for a fitting candidate;
  // the ordinary resource checks still own each final configuration's legality.
  if (!footprint || *footprint >= capabilities.getRegistersPerUnit()) return false;
  unsigned axis = source.getShape().size() - 1;
  for (unsigned other = 0; other < axis; ++other)
    if (source.getShape()[other] != result.getShape()[other] ||
        source.getAxisMaps()[other] != result.getAxisMaps()[other]) return false;
  Value coordinate = gather.getCoordinates().front();
  auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
  auto projection = queryAccessCoordinateAxes(cast<AccessOpInterface>(gather.getOperation()), 0);
  if (!coordinateType || coordinateType.getShape().size() != 1 ||
      !projection.isExact() || projection.targetToSource.size() != result.getShape().size())
    return false;
  for (auto [position, mapped] : llvm::enumerate(projection.targetToSource))
    if (mapped != (position == axis ? std::optional<unsigned>(0) : std::nullopt)) return false;
  while (auto reshape = coordinate.getDefiningOp<ReshapeOp>()) {
    auto relations = queryFragmentOperandRelations(reshape);
    auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
    if (!input || input.getShape().size() != 1 || failed(relations) ||
        !relations->front().preservesNonUnitAxes()) return false;
    coordinate = reshape.getValue();
  }
  if (auto subtract = coordinate.getDefiningOp<BinaryOp>()) {
    if (subtract.getOperatorKind() != BinaryOperator::Subtract) return false;
    auto zero = dyn_cast_or_null<IntegerAttr>(
        UniformValueAnalysis(describeUniformValue).evaluate(subtract.getRhs()));
    if (!zero || !zero.getValue().isZero()) return false;
    coordinate = subtract.getLhs();
  }
  auto range = coordinate.getDefiningOp<MakeRangeOp>();
  return range && isUnitStepRange(range) &&
      range.getResult().getType().getShape()[0] == result.getShape()[axis] &&
      IndexRelations().multipleOf(range.getStart(), range.getExtent());
}

std::optional<bool> snapshotSliceFits(GatherOp gather) {
  auto kernel = gather->getParentOfType<func::FuncOp>();
  auto source = cast<FragmentType>(gather.getSource().getType());
  auto result = cast<FragmentType>(gather.getResult().getType());
  unsigned axis = source.getShape().size() - 1;
  auto width = cast<PhysicalExprAttr>(result.getShape()[axis]);
  auto capacity = cast<PhysicalExprAttr>(source.getShape()[axis]);
  if (configurationExpressionAtMost(kernel, width, capacity)) return true;
  if (configurationExpressionLessThan(kernel, capacity, width)) return false;
  for (Operation *child = gather, *parent = child->getParentOp(); parent;
       child = parent, parent = parent->getParentOp()) {
    auto branch = dyn_cast<scf::IfOp>(parent);
    if (!branch) continue;
    auto comparison = branch.getCondition().getDefiningOp<CompareOp>();
    if (comparison && comparison.getPredicate() == ComparePredicate::Le &&
        queryLaunchExpression(comparison.getLhs()) == width &&
        queryLaunchExpression(comparison.getRhs()) == capacity)
      return child->getParentRegion() == &branch.getThenRegion();
  }
  return std::nullopt;
}

FailureOr<bool> composeLoadGatherImpl(GatherOp gather) {
  auto sourceType = dyn_cast<FragmentType>(gather.getSource().getType());
  Value loaded = gather.getSource();
  while (auto transpose = loaded.getDefiningOp<TransposeOp>()) {
    auto input = cast<FragmentType>(transpose.getValue().getType());
    auto output = transpose.getResult().getType();
    for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation())) {
      auto original = cast<AxisMapAttr>(input.getAxisMaps()[sourceAxis]);
      auto transposed = cast<AxisMapAttr>(output.getAxisMaps()[axis]);
      if (!(sourceAxisIdentity(original) == sourceAxisIdentity(transposed)) ||
          original.getDimensionId() != transposed.getDimensionId())
        return false;
    }
    loaded = transpose.getValue();
  }
  auto sourceLoad = loaded.getDefiningOp<LoadOp>();
  if (!sourceType || !sourceLoad ||
      !isa<ViewType, BufferType>(sourceLoad.getResource().getType()) ||
      !canReplayReadAt(sourceLoad, gather))
    return false;
  if (gather.getCoordinates().size() != gather.getSourceAxes().size()) {
    gather.emitOpError("gather coordinate/source-axis schema is incomplete");
    return failure();
  }

  PhysicalProgramAnalysis analysis(gather->getParentOfType<func::FuncOp>());
  OpBuilder builder(gather);
  auto resultType = dyn_cast<FragmentType>(gather.getResult().getType());
  IRMapping replay;
  SmallVector<Value> replayInputs(sourceLoad.getCoordinates());
  replayInputs.append({sourceLoad.getValid(), sourceLoad.getFill()});
  if (resultType)
    bindUnchangedAccessValues(builder, replayInputs, sourceType, resultType,
                              gather.getSourceAxes(), replay);
  // Rebuild only the selected axes. Indirect coordinates over untouched axes
  // remain their original SSA snapshots; dependent accesses still block this
  // rewrite until the composition worklist has normalized their producers.
  for (Value value : replayInputs)
    if (value && isa<FragmentType>(value.getType()) &&
        !analysis.replayAt(value, std::nullopt, PhysicalReplayScope::Coordinate,
                           /*allowAccesses=*/false, gather, replay).isReplayable())
      return false;

  SmallVector<Value> originalCoordinates;
  for (Value coordinate : sourceLoad.getCoordinates()) {
    while (auto broadcast = coordinate.getDefiningOp<BroadcastOp>())
      coordinate = broadcast.getValue();
    originalCoordinates.push_back(coordinate);
  }
  SmallVector<Value> coordinates(originalCoordinates);
  SmallVector<MakeRangeOp> selectedRanges;
  llvm::SmallDenseSet<Value> selectedRoots;
  for (int64_t sourceAxis : gather.getSourceAxes()) {
    if (sourceAxis < 0 || sourceAxis >= static_cast<int64_t>(sourceType.getShape().size()))
      return gather.emitOpError("gather source axis is outside its loaded value");
    auto root = queryExactLogicalRange(analysis.axisRanges(
        gather.getSource(), static_cast<unsigned>(sourceAxis)));
    if (failed(root) ||
        (*root).getResult().getType().getShape()[0] != sourceType.getShape()[sourceAxis] ||
        !selectedRoots.insert((*root).getResult()).second)
      return false;
    selectedRanges.push_back(*root);
  }
  for (auto [coordinate, range] :
       llvm::zip(gather.getCoordinates(), selectedRanges)) {
    // The ordinal of a retained slice often subtracts its original base.
    // Compose the inverse translation before rebuilding the access: keeping
    // start + (coordinate - start) would hide the original range's bounds.
    Value absolute = isUnitStepRange(range)
                         ? cancelIndexOffset(builder, coordinate, range.getStart())
                         : Value();
    if (absolute)
      coordinate = absolute;
    if (resultType) {
      Type element = coordinate.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      auto coordinateType = FragmentType::get(
          resultType.getContext(), element, resultType.getShape(),
          resultType.getAxisMaps(), resultType.getValidity(), resultType.getOwner());
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, gather.getLoc(), coordinate, coordinateType);
      if (failed(projected))
        return gather.emitOpError(
            "composed gather index cannot adopt its result coordinate relation");
      coordinate = *projected;
    }
    if (!absolute && (!isZero(range.getStart()) || !isUnitStepRange(range))) {
      Type indexType = range.getResult().getType().getElementType();
      if (auto fragment = dyn_cast<FragmentType>(coordinate.getType()))
        indexType = FragmentType::get(
            fragment.getContext(), indexType, fragment.getShape(),
            fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      if (coordinate.getType() != indexType)
        coordinate = builder.create<CastOp>(gather.getLoc(), indexType, coordinate);
      auto projectedBound = [&](Value bound) -> Value {
        if (auto fragment = dyn_cast<FragmentType>(indexType))
          return builder.create<BroadcastOp>(gather.getLoc(), fragment, bound);
        return bound;
      };
      // Gather indexes positions in the loaded tensor. Its ordinal must be
      // composed with the load range before becoming a resource coordinate.
      if (!isUnitStepRange(range))
        coordinate = builder.create<BinaryOp>(
            gather.getLoc(), indexType, coordinate, projectedBound(range.getStep()),
            BinaryOperator::Multiply);
      if (!isZero(range.getStart()))
        coordinate = builder.create<BinaryOp>(
            gather.getLoc(), indexType, projectedBound(range.getStart()), coordinate,
            BinaryOperator::Add);
    }
    replay.map(range.getResult(), coordinate);
  }

  if (resultType) {
    for (Value predicateOrFill : replayInputs) {
      if (!predicateOrFill)
        continue;
      PhysicalRangeFact roots = analysis.sourceRanges(predicateOrFill);
      // Bounds may use the raw range while the address uses a guarded index.
      // Preserve each untouched range's values, not the address expression.
      for (MakeRangeOp range : roots.roots) {
        if (replay.lookupOrNull(range.getResult()))
          continue;
        FailureOr<int64_t> dimension = queryRangeDimension(range);
        if (failed(dimension))
          continue;
        auto sourceAxis = queryFragmentAxis(sourceType, sourceAxisIdentity(range),
                                            *dimension);
        if (!sourceAxis.isExact()) {
          std::optional<unsigned> retainedAxis;
          bool ambiguous = false;
          for (unsigned axis = 0; axis < sourceType.getShape().size(); ++axis) {
            if (llvm::is_contained(gather.getSourceAxes(),
                                   static_cast<int64_t>(axis)))
              continue;
            auto retained = analysis.axisRanges(gather.getSource(), axis);
            if (!retained.isExact() || !retained.blockers.empty() ||
                !llvm::any_of(retained.roots, [&](MakeRangeOp root) {
                  auto rootDimension = queryRangeDimension(root);
                  return succeeded(rootDimension) && *rootDimension == *dimension &&
                         sameLogicalRange(root, range) &&
                         samePhysicalScalarExpression(root.getStart(),
                                                      range.getStart()) &&
                         samePhysicalScalarExpression(root.getExtent(),
                                                      range.getExtent());
                }))
              continue;
            if (retainedAxis) {
              ambiguous = true;
              break;
            }
            retainedAxis = axis;
          }
          if (retainedAxis && !ambiguous) {
            auto retained =
                cast<AxisMapAttr>(sourceType.getAxisMaps()[*retainedAxis]);
            sourceAxis = queryFragmentAxis(
                sourceType, sourceAxisIdentity(retained),
                retained.getDimensionId());
          }
        }
        if (!sourceAxis.isExact() ||
            llvm::is_contained(gather.getSourceAxes(),
                               static_cast<int64_t>(sourceAxis.fragmentAxis)))
          continue;
        auto retainedMapping =
            cast<AxisMapAttr>(sourceType.getAxisMaps()[sourceAxis.fragmentAxis]);
        auto axis = queryFragmentAxis(resultType, sourceAxisIdentity(retainedMapping),
                                      sourceAxis.dimensionId);
        auto rangeType = range.getResult().getType();
        if (!axis.isExact() ||
            axis.dimensionId != sourceAxis.dimensionId ||
            resultType.getShape()[axis.fragmentAxis] != rangeType.getShape()[0])
          continue;
        // Reshape can rename a retained coordinate axis. The exact range
        // dependency above supplies its result position without changing the
        // range's values or inventing an equality between unrelated axes.
        auto namedType = FragmentType::get(
            rangeType.getContext(), rangeType.getElementType(), rangeType.getShape(),
            builder.getArrayAttr({AxisMapAttr::get(
                rangeType.getContext(), retainedMapping.getSourceId(),
                retainedMapping.getSourceAxis(), retainedMapping.getDimensionId(),
                0, retainedMapping.getDerived())}),
            rangeType.getValidity(), rangeType.getOwner());
        Value retainedRange = range.getResult();
        if (namedType != rangeType) {
          auto relation = inferReshapeReassociation(rangeType, namedType);
          if (failed(relation))
            return gather.emitOpError(
                "retained coordinate has no exact renamed-axis relation");
          retainedRange = builder.create<ReshapeOp>(
              gather.getLoc(), namedType, retainedRange, *relation);
        }
        auto rangeTarget = FragmentType::get(
            resultType.getContext(), rangeType.getElementType(),
            resultType.getShape(), resultType.getAxisMaps(),
            resultType.getValidity(), resultType.getOwner());
        FailureOr<Value> projectedRange = projectPhysicalValueToSchema(
            builder, gather.getLoc(), retainedRange, rangeTarget);
        if (failed(projectedRange))
          return gather.emitOpError(
              "retained coordinate range cannot adopt the gather result relation");
        replay.map(range.getResult(), *projectedRange);
      }
    }
  }

  // A storage coordinate may combine several logical axes (for example a
  // reshaped row and column). Bind every selected range before replaying it.
  for (auto [slot, original] : llvm::enumerate(originalCoordinates)) {
    FailureOr<Value> selected = resultType
        ? replayFragmentValue(builder, original, resultType, replay, analysis, gather)
        : replayScalarValue(builder, original, replay, analysis, gather);
    if (failed(selected))
      return gather.emitOpError("indexed load coordinate cannot follow its source ranges");
    coordinates[slot] = *selected;
  }

  if (!resultType) {
    FailureOr<Value> sourceValid = replayScalarValue(
        builder, sourceLoad.getValid(), replay, analysis, gather);
    FailureOr<Value> sourceFill = replayScalarValue(
        builder, sourceLoad.getFill(), replay, analysis, gather);
    if (failed(sourceValid) || failed(sourceFill)) {
      gather.emitOpError(
          "source access validity/fill cannot follow scalar composed coordinates");
      return failure();
    }
    Value valid = *sourceValid;
    if (gather.getValid()) {
      if (!gather.getValid().getType().isInteger(1)) {
        gather.emitOpError("scalar gather validity is not scalar i1");
        return failure();
      }
      valid = valid ? Value(builder.create<BinaryOp>(
                            gather.getLoc(), builder.getI1Type(), valid,
                            gather.getValid(), BinaryOperator::LogicalAnd))
                    : gather.getValid();
    }
    Value fill = *sourceFill;
    if (sourceLoad.getValid() && gather.getValid()) {
      if (!fill || !gather.getFill() ||
          fill.getType() != gather.getResult().getType() ||
          gather.getFill().getType() != gather.getResult().getType()) {
        gather.emitOpError(
            "scalar composed conditional access has incompatible fills");
        return failure();
      }
      fill = builder.create<SelectOp>(
          gather.getLoc(), gather.getResult().getType(), gather.getValid(), fill,
          gather.getFill());
    } else if (gather.getValid()) {
      fill = gather.getFill();
    }
    if (static_cast<bool>(valid) != static_cast<bool>(fill)) {
      gather.emitOpError(
          "scalar composed access requires paired validity and fill");
      return failure();
    }
    auto replacement = builder.create<LoadOp>(
        gather.getLoc(), gather.getResult().getType(), sourceLoad.getResource(),
        coordinates, valid, fill, sourceLoad.getSourceAxes());
    if (Attribute origin = sourceLoad->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
    if (sourceLoad->getBlock() && sourceLoad.getResult().use_empty())
      sourceLoad.erase();
    return true;
  }
  FailureOr<Value> sourceValid = replayFragmentValue(
      builder, sourceLoad.getValid(), resultType, replay, analysis, gather);
  FailureOr<Value> sourceFill = replayFragmentValue(
      builder, sourceLoad.getFill(), resultType, replay, analysis, gather);
  if (failed(sourceValid) || failed(sourceFill)) {
    gather.emitOpError(
        "source access validity/fill cannot follow composed coordinates");
    return failure();
  }
  FailureOr<Value> valid = combinePredicates(
      builder, gather.getLoc(), resultType, *sourceValid, gather.getValid());
  if (failed(valid)) {
    gather.emitOpError(
        "source and gather validity cannot share the composed result relation");
    return failure();
  }
  Value fill = *sourceFill;
  if (sourceLoad.getValid() && gather.getValid()) {
    Value gatherFill = gather.getFill();
    if (!fill || !gatherFill) {
      gather.emitOpError(
          "composed conditional access requires both source and gather fill");
      return failure();
    }
    auto result = resultType;
    if (fill.getType() != result)
      fill = builder.create<BroadcastOp>(gather.getLoc(), result, fill);
    if (gatherFill.getType() != result)
      gatherFill =
          builder.create<BroadcastOp>(gather.getLoc(), result, gatherFill);
    Value condition = gather.getValid();
    auto predicate = FragmentType::get(
        result.getContext(), builder.getI1Type(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
    if (condition.getType() != predicate)
      condition =
          builder.create<BroadcastOp>(gather.getLoc(), predicate, condition);
    fill = builder.create<SelectOp>(gather.getLoc(), result, condition, fill,
                                    gatherFill);
  } else if (gather.getValid()) {
    fill = gather.getFill();
  }
  auto replacement = builder.create<LoadOp>(
      gather.getLoc(), gather.getResult().getType(), sourceLoad.getResource(),
      coordinates, *valid, fill,
      sourceLoad.getSourceAxes());
  if (Attribute origin = sourceLoad->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  if (sourceLoad->getBlock() && sourceLoad.getResult().use_empty())
    sourceLoad.erase();
  return true;
}

} // namespace

FailureOr<bool> composeLoadGather(GatherOp gather) {
  if (!isAlignedSnapshotSlice(gather)) return composeLoadGatherImpl(gather);
  auto fits = snapshotSliceFits(gather);
  if (fits) return *fits ? FailureOr<bool>(false) : composeLoadGatherImpl(gather);

  // The complete source is already required by another computation. Preserve
  // that snapshot for aligned shrinking sub-tiles, without changing the load's
  // mask/fill or extending its coverage. Coverage and tile parameters are bound
  // at invocation; the alternative remains the original access composition.
  OpBuilder builder(gather);
  Location location = gather.getLoc();
  auto source = cast<FragmentType>(gather.getSource().getType());
  auto result = cast<FragmentType>(gather.getResult().getType());
  unsigned axis = source.getShape().size() - 1;
  auto width = builder.create<PhysicalExprOp>(location, builder.getIndexType(),
      cast<PhysicalExprAttr>(result.getShape()[axis]));
  auto capacity = builder.create<PhysicalExprOp>(location, builder.getIndexType(),
      cast<PhysicalExprAttr>(source.getShape()[axis]));
  auto condition = builder.create<CompareOp>(location, builder.getI1Type(), width,
                                            capacity, ComparePredicate::Le);
  auto choice = builder.create<scf::IfOp>(location, TypeRange{result}, condition, true);
  OpBuilder retained = choice.getThenBodyBuilder();
  auto selected = cast<GatherOp>(retained.clone(*gather.getOperation()));
  retained.create<scf::YieldOp>(location, selected.getResult());
  OpBuilder composed = choice.getElseBodyBuilder();
  auto original = cast<GatherOp>(composed.clone(*gather.getOperation()));
  composed.create<scf::YieldOp>(location, original.getResult());
  auto changed = composeLoadGatherImpl(original);
  if (failed(changed) || !*changed) {
    choice.erase();
    condition.erase();
    capacity.erase();
    width.erase();
    return failed(changed) ? FailureOr<bool>(failure()) : composeLoadGatherImpl(gather);
  }
  gather.getResult().replaceAllUsesWith(choice.getResult(0));
  gather.erase();
  return true;
}

namespace {

bool sameReadCoordinates(LoadOp available, LoadOp current) {
  if (!isa<ViewType>(current.getResource().getType()) ||
      available.getResource() != current.getResource() ||
      available.getResult().getType() != current.getResult().getType() ||
      available.getSourceAxes() != current.getSourceAxes() ||
      available.getCoordinates().size() != current.getCoordinates().size())
    return false;
  return llvm::all_of(llvm::zip(available.getCoordinates(), current.getCoordinates()),
      [](auto pair) {
        auto [left, right] = pair;
        return left == right || (left.getType().isIndex() &&
            right.getType().isIndex() && IndexRelations().same(left, right));
      });
}

// Exact boundary forms describe the same resource coordinates, not merely
// masks with similar syntax. Bounds must close on both accesses, so a lower
// bound and an upper bound on one axis cannot be mistaken for equal validity.
std::optional<bool> coveredRead(LoadOp available, LoadOp current,
                                PhysicalProgramAnalysis &analysis) {
  if (!sameReadCoordinates(available, current)) return std::nullopt;
  bool equalValidity = available.getValid() == current.getValid();
  if (!equalValidity && available.getValid()) {
    auto earlier = analysis.boundaryValidity(available);
    auto later = analysis.boundaryValidity(current);
    if (!earlier.isExact() || !later.isExact() ||
        !earlier.rangeBounds.empty() || !later.rangeBounds.empty() ||
        !analysis.accessBounds(available).isExact() ||
        !analysis.accessBounds(current).isExact() ||
        !llvm::all_of(earlier.boundaryAxes, [&](int64_t axis) {
          return llvm::is_contained(later.boundaryAxes, axis);
        })) return std::nullopt;
    equalValidity = earlier.boundaryAxes == later.boundaryAxes;
  }
  bool equalFill = available.getFill() == current.getFill();
  if (!equalFill && available.getFill() && current.getFill()) {
    UniformValueAnalysis uniform(describeUniformValue);
    Attribute earlier = uniform.evaluate(available.getFill());
    Attribute later = uniform.evaluate(current.getFill());
    // Attribute equality retains floating-point payload bits and signed zero.
    equalFill = earlier && earlier == later;
  }
  return !current.getValid() || (equalValidity && equalFill);
}

} // namespace

bool reuseStableLoads(func::FuncOp kernel) {
  SmallVector<LoadOp> loads;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  bool changed = false;
  for (LoadOp current : loads) {
    if (!current->getBlock())
      continue;
    PhysicalProgramAnalysis analysis(kernel);
    Block *block = current->getBlock();
    auto cursor = current->getIterator();
    while (cursor != block->begin()) {
      --cursor;
      Operation *candidate = &*cursor;
      if (auto available = dyn_cast<LoadOp>(candidate)) {
        auto direct = coveredRead(available, current, analysis);
        if (!direct)
          continue;
        if (!canReplayReadAt(available, current))
          break;
        Value replacement = available.getResult();
        if (!*direct) {
          OpBuilder builder(current);
          replacement = builder.create<SelectOp>(current.getLoc(), current.getType(),
              current.getValid(), replacement, current.getFill());
          if (Attribute origin = current->getAttr(originAttr))
            replacement.getDefiningOp()->setAttr(originAttr, origin);
        }
        current.getResult().replaceAllUsesWith(replacement);
        current.erase();
        changed = true;
        break;
      }
      // Matching reads use the shared alias-aware snapshot proof above.
    }
  }
  return changed;
}

void sinkStableLoadChains(func::FuncOp kernel) {
  llvm::SmallPtrSet<Operation *, 32> selected;
  SmallVector<Operation *> pending;
  kernel.walk([&](LoadOp load) {
    auto view = dyn_cast<ViewType>(load.getResource().getType());
    if (view && view.getAccess() == 0 && selected.insert(load).second)
      pending.push_back(load);
  });
  while (!pending.empty()) {
    Operation *producer = pending.pop_back_val();
    for (Operation *user : producer->getUsers()) {
      if (user->getBlock() != producer->getBlock() ||
          !placement::isMovableValueOperation(user) ||
          !selected.insert(user).second)
        continue;
      pending.push_back(user);
    }
  }
  SmallVector<Operation *> ordered;
  kernel.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (selected.contains(operation))
      ordered.push_back(operation);
  });
  for (Operation *operation : llvm::reverse(ordered)) {
    Operation *firstUse = nullptr;
    for (Operation *user : operation->getUsers()) {
      Operation *ancestor = operation->getBlock()->findAncestorOpInBlock(*user);
      if (!ancestor) {
        firstUse = nullptr;
        break;
      }
      if (!firstUse || ancestor->isBeforeInBlock(firstUse))
        firstUse = ancestor;
    }
    if (!firstUse || operation->getNextNode() == firstUse)
      continue;
    if (placement::canMoveBefore(operation, firstUse))
      operation->moveBefore(firstUse);
  }
}

} // namespace intent::gpu::access
