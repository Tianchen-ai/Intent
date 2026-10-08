#include "AccessComposition.h"
#include "../Value/ReplayPolicy.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;

namespace intent::gpu::access {

namespace {

bool hasNonUnitAxisSplit(const FragmentOperandRelation &relation) {
  auto source = cast<FragmentType>(relation.sourceType);
  auto result = cast<FragmentType>(relation.resultType);
  if (source.getShape().size() >= result.getShape().size())
    return false;
  bool split = false;
  for (const FragmentAxisGroup &group : relation.groups) {
    if (group.resultAxes.empty())
      return false;
    if (group.sourceAxes.empty() && !unitAxes(result, group.resultAxes))
      return false;
    if (group.sourceAxes.size() == 1 && group.resultAxes.size() > 1) {
      for (unsigned axis : group.resultAxes) {
        auto extent = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (extent.getKind() !=
                PhysicalExprKind::Constant ||
            extent.getValue() <= 1)
          return false;
      }
      split = true;
    }
  }
  return split;
}

} // namespace

bool composeReshapedPointwise(ReshapeOp reshape) {
  auto source = cast<FragmentType>(reshape.getValue().getType());
  auto result = cast<FragmentType>(reshape.getResult().getType());
  if (auto inner = reshape.getValue().getDefiningOp<ReshapeOp>();
      inner && inner.getValue().getType() == result &&
      inner.getReassociation().size() == reshape.getReassociation().size() &&
      llvm::all_of(llvm::zip(inner.getReassociation(), reshape.getReassociation()),
                   [](auto pair) {
                     auto first = cast<ReshapeGroupAttr>(std::get<0>(pair));
                     auto second = cast<ReshapeGroupAttr>(std::get<1>(pair));
                     return first.getSourceAxes() == second.getResultAxes() &&
                            first.getResultAxes() == second.getSourceAxes();
                   })) {
    reshape.getResult().replaceAllUsesWith(inner.getValue());
    reshape.erase();
    return true;
  }
  auto relations = queryFragmentOperandRelations(reshape);
  if (failed(relations))
    return false;
  if (!hasNonUnitAxisSplit(relations->front()) &&
      source.getShape().size() <= result.getShape().size())
    return false;
  Operation *producer = reshape.getValue().getDefiningOp();
  if (!producer ||
      !isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
           SplatOp, BroadcastOp>(producer))
    return false;
  // Push the bijective shape change through computation only when that
  // computation is replaced. A shared result is an existing complete snapshot;
  // retain its reshape instead of duplicating the numerical producer DAG.
  if (!reshape.getValue().hasOneUse()) return false;
  for (Value operand : producer->getOperands()) {
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (fragment &&
        (fragment.getShape() != source.getShape() ||
         fragment.getAxisMaps() != source.getAxisMaps() ||
         fragment.getOwner() != source.getOwner()))
      return false;
  }
  OpBuilder builder(reshape);
  IRMapping mapping;
  SmallVector<Operation *> projectedOperands;
  for (Value operand : producer->getOperands()) {
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (!fragment || mapping.contains(operand))
      continue;
    auto target = FragmentType::get(
        result.getContext(), fragment.getElementType(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
    auto projected = builder.create<ReshapeOp>(
        reshape.getLoc(), target, operand, reshape.getReassociation());
    projectedOperands.push_back(projected);
    mapping.map(operand, projected.getResult());
  }
  auto replacement = cloneWithSchema(builder, producer, mapping, TypeRange{result});
  if (failed(replacement)) {
    for (Operation *projected : llvm::reverse(projectedOperands)) projected->erase();
    return false;
  }
  reshape.getResult().replaceAllUsesWith(replacement->front());
  reshape.erase();
  return true;
}

FailureOr<bool> composeReshapedLoad(ReshapeOp reshape) {
  Value sourceValue = reshape.getValue();
  Value loaded = sourceValue;
  bool transposed = false;
  while (true) {
    if (auto broadcast = loaded.getDefiningOp<BroadcastOp>()) {
      auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
      auto output = dyn_cast<FragmentType>(broadcast.getResult().getType());
      if (!input || !output || input.getShape() != output.getShape())
        break;
      auto relations = queryFragmentOperandRelations(broadcast);
      if (failed(relations) || llvm::any_of(
              relations->front().groups, [](const FragmentAxisGroup &group) {
                return group.sourceAxes.size() != 1 || group.resultAxes.size() != 1 ||
                       group.sourceAxes.front() != group.resultAxes.front();
              }))
        break;
      loaded = broadcast.getValue();
      continue;
    }
    auto transpose = loaded.getDefiningOp<TransposeOp>();
    if (!transpose)
      break;
    auto source = cast<FragmentType>(transpose.getValue().getType());
    auto result = cast<FragmentType>(transpose.getResult().getType());
    auto relations = queryFragmentOperandRelations(transpose);
    if (failed(relations))
      return false;
    for (const FragmentAxisGroup &group : relations->front().groups) {
      auto original = cast<AxisMapAttr>(source.getAxisMaps()[group.sourceAxes.front()]);
      auto transposed = cast<AxisMapAttr>(result.getAxisMaps()[group.resultAxes.front()]);
      if (!(sourceAxisIdentity(original) == sourceAxisIdentity(transposed)) ||
          original.getDimensionId() != transposed.getDimensionId())
        return false;
    }
    transposed = true;
    loaded = transpose.getValue();
  }
  auto load = loaded.getDefiningOp<LoadOp>();
  auto result = dyn_cast<FragmentType>(reshape.getResult().getType());
  if (!load || !result || !isa<ViewType>(load.getResource().getType()) ||
      !canReplayReadAt(load, reshape))
    return false;
  auto source = cast<FragmentType>(sourceValue.getType());
  auto relations = queryFragmentOperandRelations(reshape);
  if (failed(relations))
    return false;
  const auto &groups = relations->front().groups;
  if (source.getShape().size() <= result.getShape().size() &&
      !hasNonUnitAxisSplit(relations->front()) && !transposed)
    return false;
  auto kernel = reshape->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  // This access rewrite owns the whole load coordinate frame; execution-prefix
  // axes are handled by their existing traversal, outside this policy's scope.
  if (llvm::any_of(groups, [](const FragmentAxisGroup &group) {
        return group.kind != FragmentAxisRelationKind::Reassociation;
      }))
    return false;
  unsigned sourceRank = source.getShape().size();
  unsigned resultRank = result.getShape().size();
  for (const FragmentAxisGroup &group : groups) {
    if (group.resultAxes.empty())
      for (unsigned axis : group.sourceAxes) {
        auto extent = cast<PhysicalExprAttr>(source.getShape()[axis]);
        PhysicalRangeFact ranges = analysis.axisRanges(sourceValue, axis);
        if (extent.getKind() !=
                PhysicalExprKind::Constant ||
            extent.getValue() != 1 || !ranges.isExact() ||
            !llvm::all_of(ranges.roots, isProvablySingletonLogicalRange))
          return false;
      }
    if (group.sourceAxes.empty())
      for (unsigned axis : group.resultAxes) {
        auto extent = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (extent.getKind() !=
                PhysicalExprKind::Constant ||
            extent.getValue() != 1)
          return false;
      }
  }
  SmallVector<bool> preservedSource(sourceRank, false);
  SmallVector<bool> preservedResult(resultRank, false);
  for (const FragmentAxisGroup &group : groups) {
    if (group.sourceAxes.empty()) {
      for (unsigned axis : group.resultAxes)
        preservedResult[axis] = true;
      continue;
    }
    if (group.resultAxes.empty()) {
      for (unsigned axis : group.sourceAxes)
        preservedSource[axis] = true;
      continue;
    }
    if (group.sourceAxes.size() != 1 || group.resultAxes.size() != 1)
      continue;
    unsigned sourceAxis = group.sourceAxes.front();
    unsigned resultAxis = group.resultAxes.front();
    auto original = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    auto target = cast<AxisMapAttr>(result.getAxisMaps()[resultAxis]);
    if (sourceAxisIdentity(original) == sourceAxisIdentity(target) &&
        original.getDimensionId() == target.getDimensionId() &&
        source.getShape()[sourceAxis] == result.getShape()[resultAxis]) {
      preservedSource[sourceAxis] = true;
      preservedResult[resultAxis] = true;
    }
  }
  SmallVector<MakeRangeOp> sourceRanges(sourceRank);
  SmallVector<SmallVector<MakeRangeOp>> sourceRoots(sourceRank);
  SmallVector<Attribute> sourceExtents(sourceRank);
  llvm::DenseMap<Operation *, unsigned> rootAxes;
  for (unsigned axis = 0; axis < sourceRank; ++axis) {
    PhysicalRangeFact fact = analysis.axisRanges(sourceValue, axis);
    for (MakeRangeOp root : fact.roots) {
      auto [found, inserted] = rootAxes.try_emplace(root.getOperation(), axis);
      if (!inserted && found->second != axis)
        return false;
    }
    sourceRoots[axis].append(fact.roots.begin(), fact.roots.end());
    if (preservedSource[axis]) {
      auto realization = analysis.axisRealization(sourceValue, axis);
      bool introducedUnit = fact.roots.empty() && realization.isExact() &&
                            !realization.constructionScalarSeed &&
                            cast<PhysicalExprAttr>(source.getShape()[axis]).getKind() ==
                                PhysicalExprKind::Constant &&
                            cast<PhysicalExprAttr>(source.getShape()[axis]).getValue() == 1;
      if (fact.state != PhysicalFactState::Exact && !introducedUnit)
        return false;
      continue;
    }
    FailureOr<MakeRangeOp> authority = queryExactLogicalRange(fact);
    if (failed(authority))
      return false;
    for (MakeRangeOp root : fact.roots) {
      if (!samePhysicalScalarExpression(root.getStart(),
                                         root.getLogicalStart()) ||
          !isUnitStepRange(root))
        return false;
      auto count = constantLogicalRangeCardinality(root);
      auto physical =
          cast<PhysicalExprAttr>(root.getResult().getType().getShape()[0]);
      bool covered = count &&
                     physical.getKind() ==
                         PhysicalExprKind::Constant &&
                     physical.getValue() >= *count;
      auto realization = analysis.axisRealization(root.getResult(), 0);
      if (!covered &&
          (!isZero(root.getLogicalStart()) ||
           (!realization.constructionScalarSeed &&
            !samePhysicalScalarExpression(root.getExtent(),
                                           root.getLogicalStop()))))
        return false;
    }
    MakeRangeOp range = *authority;
    auto count = constantLogicalRangeCardinality(range);
    PhysicalExprAttr extent =
        count ? PhysicalExprAttr::get(
                    reshape.getContext(),
                    PhysicalExprKind::Constant, *count,
                    StringAttr::get(reshape.getContext(), ""),
                    ArrayAttr::get(reshape.getContext(), {}))
              : queryLaunchExpression(range.getLogicalStop());
    if (!extent)
      return false;
    if (!count && !queryNonNegativeIndexUpperBound(range.getLogicalStop())) {
      auto zero = PhysicalExprAttr::get(
          reshape.getContext(), PhysicalExprKind::Constant,
          0, StringAttr::get(reshape.getContext(), ""), ArrayAttr::get(reshape.getContext(), {}));
      extent = PhysicalExprAttr::get(
          reshape.getContext(), PhysicalExprKind::Maximum,
          0, StringAttr::get(reshape.getContext(), ""),
          ArrayAttr::get(reshape.getContext(), {extent, zero}));
    }
    sourceRanges[axis] = range;
    sourceExtents[axis] = extent;
  }
  SmallVector<PhysicalExprAttr> resultExtents(resultRank);
  for (const FragmentAxisGroup &group : groups) {
    if (group.resultAxes.empty())
      continue;
    unsigned resultAxis = group.resultAxes.front();
    if (preservedResult[resultAxis])
      continue;
    ArrayRef<unsigned> axes = group.sourceAxes;
    auto extent = cast<PhysicalExprAttr>(sourceExtents[axes.front()]);
    if (group.resultAxes.size() > 1) {
      // The declared row-major group may split a complete source axis.
      // Static result extents give each constituent its own exact range;
      // physical padding or a partial source tile cannot satisfy this proof.
      int64_t product = 1;
      for (unsigned axis : group.resultAxes) {
        auto part = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (part.getKind() !=
                PhysicalExprKind::Constant ||
            part.getValue() <= 0 ||
            product > std::numeric_limits<int64_t>::max() / part.getValue())
          return false;
        product *= part.getValue();
        resultExtents[axis] = part;
      }
      int64_t sourceProduct = 1;
      for (unsigned axis : axes) {
        auto part = cast<PhysicalExprAttr>(sourceExtents[axis]);
        if (part.getKind() != PhysicalExprKind::Constant ||
            part.getValue() <= 0 ||
            sourceProduct > std::numeric_limits<int64_t>::max() / part.getValue())
          return false;
        sourceProduct *= part.getValue();
      }
      if (sourceProduct != product)
        return false;
      continue;
    }
    for (unsigned axis : axes.drop_front())
      extent = PhysicalExprAttr::get(reshape.getContext(),
          PhysicalExprKind::Multiply, 0,
          StringAttr::get(reshape.getContext(), ""),
          ArrayAttr::get(reshape.getContext(), {extent, sourceExtents[axis]}));
    resultExtents[resultAxis] = extent;
  }
  if (llvm::all_of(preservedResult, [](bool preserved) { return preserved; })) {
    bool exposesReductionPairs = llvm::any_of(
        reshape.getResult().getUsers(), [&](Operation *user) {
          auto contract = dyn_cast<ContractOp>(user);
          return contract && contract.getLhsReductionAxes().size() > 1 &&
                 (contract.getLhs() == reshape.getResult() ||
                  contract.getRhs() == reshape.getResult());
        });
    if (transposed || !exposesReductionPairs)
      return false;
    // Unit-axis insertion/removal changes the access schema, not its members.
    // Expose the load when this enables multi-pair contraction normalization;
    // a single-pair contraction can retain its existing load factorization.
    OpBuilder builder(reshape);
    auto project = [&](Value value) -> FailureOr<Value> {
      if (!value)
        return Value();
      auto fragment = dyn_cast<FragmentType>(value.getType());
      if (!fragment)
        return value;
      auto expandedType = FragmentType::get(
          source.getContext(), fragment.getElementType(), source.getShape(),
          source.getAxisMaps(), source.getValidity(), source.getOwner());
      auto expanded = projectPhysicalValueToSchema(
          builder, reshape.getLoc(), value, expandedType);
      if (failed(expanded))
        return failure();
      auto projectedType = FragmentType::get(
          result.getContext(), fragment.getElementType(), result.getShape(),
          result.getAxisMaps(), result.getValidity(), result.getOwner());
      return Value(builder.create<ReshapeOp>(
          reshape.getLoc(), projectedType, *expanded,
          reshape.getReassociation()));
    };
    SmallVector<Value> coordinates;
    for (Value coordinate : load.getCoordinates()) {
      auto projected = project(coordinate);
      if (failed(projected))
        return reshape.emitOpError("cannot project a unit-axis load coordinate"),
               failure();
      coordinates.push_back(*projected);
    }
    auto valid = project(load.getValid());
    auto fill = project(load.getFill());
    if (failed(valid) || failed(fill))
      return reshape.emitOpError("cannot project unit-axis load validity/fill"),
             failure();
    auto replacement = builder.create<LoadOp>(
        reshape.getLoc(), result, load.getResource(), coordinates, *valid, *fill,
        load.getSourceAxesAttr());
    replacement->setDiscardableAttrs(
        llvm::to_vector(load->getDiscardableAttrs()));
    reshape.getResult().replaceAllUsesWith(replacement.getResult());
    reshape.erase();
    return true;
  }
  OpBuilder builder(reshape);
  auto materializeExtent = [&](PhysicalExprAttr extent) -> Value {
    if (extent.getKind() ==
        PhysicalExprKind::Constant)
      return builder.create<arith::ConstantIndexOp>(reshape.getLoc(),
                                                    extent.getValue());
    return builder.create<PhysicalExprOp>(reshape.getLoc(), builder.getIndexType(),
                                         extent);
  };
  SmallVector<Value> resultStops(resultRank);
  for (unsigned axis = 0; axis < resultRank; ++axis)
    if (!preservedResult[axis])
      resultStops[axis] = materializeExtent(resultExtents[axis]);

  Value zero = builder.create<arith::ConstantIndexOp>(reshape.getLoc(), 0);
  Value one = builder.create<arith::ConstantIndexOp>(reshape.getLoc(), 1);
  SmallVector<Value> flatCoordinates(resultRank);
  auto indexType = FragmentType::get(
      result.getContext(), builder.getIndexType(), result.getShape(),
      result.getAxisMaps(), result.getValidity(), result.getOwner());
  for (unsigned axis = 0; axis < resultRank; ++axis) {
    if (preservedResult[axis])
      continue;
    auto mapping = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
    Value stop = resultStops[axis];
    Value extent = materializeExtent(
        cast<PhysicalExprAttr>(result.getShape()[axis]));
    auto rangeType = FragmentType::get(
        result.getContext(), builder.getIndexType(),
        builder.getArrayAttr({result.getShape()[axis]}),
        builder.getArrayAttr({AxisMapAttr::get(
            result.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), 0, mapping.getDerived())}),
        result.getValidity(), result.getOwner());
    Value range = builder.create<MakeRangeOp>(
        reshape.getLoc(), rangeType, zero, extent, one, zero, stop,
        mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, reshape.getLoc(), range, indexType);
    if (failed(projected))
      return reshape.emitOpError("collapsed load has no flat coordinate projection");
    flatCoordinates[axis] = *projected;
  }
  SmallVector<Value> sourceCoordinates(sourceRank);
  IRMapping mapping;
  for (const FragmentAxisGroup &group : groups) {
    if (group.sourceAxes.empty())
      continue;
    if (group.resultAxes.empty()) {
      // A removed logical singleton still contributes its original address.
      // It becomes uniform in the destination shape, not an extra flat axis.
      for (unsigned axis : group.sourceAxes)
        for (MakeRangeOp root : sourceRoots[axis])
          mapping.map(root.getResult(), builder.create<SplatOp>(
              reshape.getLoc(), indexType, root.getStart()).getResult());
      continue;
    }
    if (preservedResult[group.resultAxes.front()]) {
      // Unmerged axes retain their current tile and coordinates, including
      // program-local batch coordinates and already blocked free dimensions.
      for (MakeRangeOp root : sourceRoots[group.sourceAxes.front()]) {
        unsigned axis = group.resultAxes.front();
        SmallVector<Attribute> shape(resultRank,
            PhysicalExprAttr::get(result.getContext(),
                PhysicalExprKind::Constant, 1,
                builder.getStringAttr(""), builder.getArrayAttr({})));
        shape[axis] = root.getResult().getType().getShape()[0];
        SmallVector<Attribute> groups;
        for (unsigned position = 0; position < resultRank; ++position) {
          SmallVector<int64_t> inputAxes;
          if (position == axis)
            inputAxes.push_back(0);
          groups.push_back(ReshapeGroupAttr::get(result.getContext(),
              builder.getDenseI64ArrayAttr(inputAxes),
              builder.getDenseI64ArrayAttr({position})));
        }
        auto shaped = FragmentType::get(result.getContext(), builder.getIndexType(),
            builder.getArrayAttr(shape), indexType.getAxisMaps(),
            indexType.getValidity(), indexType.getOwner());
        Value positioned = builder.create<ReshapeOp>(reshape.getLoc(), shaped,
            root.getResult(), builder.getArrayAttr(groups));
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, reshape.getLoc(), positioned, indexType);
        if (failed(projected))
          return reshape.emitOpError("collapsed load lost an unmerged axis projection");
        mapping.map(root.getResult(), *projected);
      }
      continue;
    }
    Value ordinal = flatCoordinates[group.resultAxes.front()];
    for (unsigned axis : ArrayRef<unsigned>(group.resultAxes).drop_front()) {
      Value extent = builder.create<SplatOp>(reshape.getLoc(), indexType,
                                            resultStops[axis]);
      ordinal = builder.create<BinaryOp>(reshape.getLoc(), indexType, ordinal,
                                        extent, BinaryOperator::Multiply);
      ordinal = builder.create<BinaryOp>(reshape.getLoc(), indexType, ordinal,
                                        flatCoordinates[axis],
                                        BinaryOperator::Add);
    }
    ArrayRef<unsigned> axes = group.sourceAxes;
    for (unsigned position = axes.size(); position-- > 0;) {
      unsigned axis = axes[position];
      MakeRangeOp range = sourceRanges[axis];
      Value coordinate = ordinal;
      if (position != 0) {
        Value divisor = builder.create<BinaryOp>(
            reshape.getLoc(), builder.getIndexType(),
            materializeExtent(cast<PhysicalExprAttr>(sourceExtents[axis])), one,
            BinaryOperator::Maximum);
        Value extent = builder.create<SplatOp>(reshape.getLoc(), indexType, divisor);
        coordinate = builder.create<BinaryOp>(
            reshape.getLoc(), indexType, ordinal, extent, BinaryOperator::Remainder);
        ordinal = builder.create<BinaryOp>(
            reshape.getLoc(), indexType, ordinal, extent, BinaryOperator::FloorDivide);
      }
      if (!isZero(range.getLogicalStart())) {
        Value start = builder.create<SplatOp>(
            reshape.getLoc(), indexType, range.getLogicalStart());
        coordinate = builder.create<BinaryOp>(
            reshape.getLoc(), indexType, start, coordinate, BinaryOperator::Add);
      }
      sourceCoordinates[axis] = coordinate;
      for (MakeRangeOp root : sourceRoots[axis])
        mapping.map(root.getResult(), coordinate);
    }
  }
  SmallVector<Value> coordinates;
  for (Value coordinate : load.getCoordinates()) {
    FailureOr<Value> replayed = replayFragmentValue(
        builder, coordinate, result, mapping, analysis, reshape);
    if (failed(replayed))
      return reshape.emitOpError("collapsed load could not preserve its coordinate graph");
    coordinates.push_back(*replayed);
  }
  FailureOr<Value> valid = replayFragmentValue(
      builder, load.getValid(), result, mapping, analysis, reshape);
  FailureOr<Value> fill = replayFragmentValue(
      builder, load.getFill(), result, mapping, analysis, reshape);
  if (failed(valid) || failed(fill))
    return reshape.emitOpError("collapsed load could not preserve validity and fill");
  auto predicate = FragmentType::get(
      result.getContext(), builder.getI1Type(), result.getShape(), result.getAxisMaps(),
      result.getValidity(), result.getOwner());
  Value active = *valid;
  for (unsigned axis = 0; axis < sourceRank; ++axis) {
    if (preservedSource[axis])
      continue;
    Value coordinate = sourceCoordinates[axis];
    Value lower = builder.create<SplatOp>(
        reshape.getLoc(), indexType, sourceRanges[axis].getLogicalStart());
    Value end = builder.create<SplatOp>(
        reshape.getLoc(), indexType, sourceRanges[axis].getLogicalStop());
    Value nonNegative = builder.create<CompareOp>(
        reshape.getLoc(), predicate, coordinate, lower, ComparePredicate::Ge);
    Value belowEnd = builder.create<CompareOp>(
        reshape.getLoc(), predicate, coordinate, end, ComparePredicate::Lt);
    Value within = builder.create<BinaryOp>(
        reshape.getLoc(), predicate, nonNegative, belowEnd, BinaryOperator::LogicalAnd);
    active = active ? Value(builder.create<BinaryOp>(
        reshape.getLoc(), predicate, active, within, BinaryOperator::LogicalAnd)) : within;
  }
  if (!*fill) {
    fill = materializeZeroFragment(builder, reshape.getLoc(), result);
    if (failed(fill))
      return failure();
  }
  auto replacement = builder.create<LoadOp>(
      reshape.getLoc(), result, load.getResource(), coordinates, active, *fill,
      load.getSourceAxes());
  if (Attribute origin = load->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  reshape.getResult().replaceAllUsesWith(replacement.getResult());
  reshape.erase();
  if (load.getResult().use_empty())
    load.erase();
  return true;
}

FailureOr<bool> composeReshapedStore(StoreOp store) {
  auto isPositionalRebinding = [](Operation *operation) {
      if (!isa_and_nonnull<BroadcastOp, ReshapeOp>(operation))
        return false;
      auto source = dyn_cast<FragmentType>(operation->getOperand(0).getType());
      auto result = dyn_cast<FragmentType>(operation->getResult(0).getType());
      if (!source || !result || source.getShape() != result.getShape() ||
          source.getElementType() != result.getElementType() ||
          source.getOwner() != result.getOwner() || source.getValidity() != result.getValidity())
        return false;
      auto relations = queryFragmentOperandRelations(operation);
      if (failed(relations))
        return false;
      for (const FragmentAxisGroup &group : relations->front().groups) {
        if (group.resultAxes.size() != 1 || group.sourceAxes.size() > 1)
          return false;
        unsigned axis = group.resultAxes.front();
        if (!group.sourceAxes.empty()) {
          if (group.sourceAxes.front() != axis)
            return false;
          continue;
        }
        if (isa<ReshapeOp>(operation))
          return false;
        auto extent = cast<PhysicalExprAttr>(source.getShape()[axis]);
        bool unit = extent.getKind() ==
                        PhysicalExprKind::Constant &&
                    extent.getValue() == 1;
        if (!unit)
          return false;
      }
      return true;
  };
  auto stripPositionalRebindings = [&](Value value) {
    while (isPositionalRebinding(value.getDefiningOp()))
      value = value.getDefiningOp()->getOperand(0);
    return value;
  };
  auto output = dyn_cast<FragmentType>(store.getValue().getType());
  if (!output || output.getShape().empty())
    return false;
  auto kernel = store->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  unsigned outputRank = output.getShape().size();
  SmallVector<MakeRangeOp> outputRanges(outputRank);
  SmallVector<unsigned> coordinateSlots(outputRank);
  for (auto [slot, coordinate] : llvm::enumerate(store.getCoordinates())) {
    if (!isa<FragmentType>(coordinate.getType()))
      continue;
    auto range = coordinate.getDefiningOp<MakeRangeOp>();
    FailureOr<int64_t> dimension = range ? queryRangeDimension(range)
                                         : FailureOr<int64_t>(failure());
    if (!range || failed(dimension) || !isZero(range.getStart()) ||
        !isZero(range.getLogicalStart()) || !isUnitStepRange(range))
      return false;
    auto realization = analysis.axisRealization(range.getResult(), 0);
    if (!realization.constructionScalarSeed &&
        !samePhysicalScalarExpression(range.getExtent(), range.getLogicalStop()))
      return false;
    auto projection = queryFragmentDimension(output, *dimension);
    if (!projection.isExact() || outputRanges[projection.fragmentAxis])
      return false;
    outputRanges[projection.fragmentAxis] = range;
    coordinateSlots[projection.fragmentAxis] = slot;
  }
  if (llvm::any_of(outputRanges, [](MakeRangeOp range) { return !range; }))
    return false;

  SmallVector<unsigned> outerAxes;
  for (unsigned axis = 0; axis < outputRank; ++axis)
    outerAxes.push_back(axis);
  auto isPointwise = [](Operation *operation) {
    return operation && isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
                            BitcastOp, SplatOp, BroadcastOp>(operation);
  };
  ReshapeOp reshape;
  llvm::SmallPtrSet<Operation *, 8> transposes;
  std::function<bool(Value, SmallVector<unsigned>)> findReshape =
      [&](Value value, SmallVector<unsigned> axes) {
        value = stripPositionalRebindings(value);
        if (auto transpose = value.getDefiningOp<TransposeOp>()) {
          auto source = cast<FragmentType>(transpose.getValue().getType());
          auto result = cast<FragmentType>(transpose.getResult().getType());
          auto relations = queryFragmentOperandRelations(transpose);
          if (failed(relations) || result.getShape().size() != axes.size())
            return false;
          SmallVector<unsigned> inputAxes(axes.size());
          for (const FragmentAxisGroup &group : relations->front().groups) {
            unsigned axis = group.resultAxes.front();
            unsigned sourceAxis = group.sourceAxes.front();
            auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
            auto resultMap = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
            if (!(sourceAxisIdentity(sourceMap) == sourceAxisIdentity(resultMap)) ||
                sourceMap.getDimensionId() != resultMap.getDimensionId())
              return false;
            inputAxes[sourceAxis] = axes[axis];
          }
          if (!findReshape(transpose.getValue(), std::move(inputAxes)))
            return false;
          transposes.insert(transpose);
          return true;
        }
        if (auto candidate = value.getDefiningOp<ReshapeOp>()) {
          reshape = candidate;
          outerAxes = std::move(axes);
          return true;
        }
        Operation *producer = value.getDefiningOp();
        return isPointwise(producer) && !isa<BroadcastOp>(producer) &&
               llvm::any_of(producer->getOperands(), [&](Value operand) {
                 return findReshape(operand, axes);
               });
      };
  if (!findReshape(store.getValue(), outerAxes))
    return false;
  SmallVector<LoadOp> companions;
  SmallVector<ReshapeOp> sourceReshapes{reshape};
  llvm::DenseMap<Value, unsigned> companionAxes;
  using OutputAxes = SmallVector<std::optional<unsigned>>;
  llvm::DenseMap<Value, OutputAxes> checked;
  auto isSourceFrame = [&](ArrayRef<std::optional<unsigned>> axes) {
    return axes.size() == outerAxes.size() &&
           llvm::all_of(llvm::zip(axes, outerAxes), [](const auto &pair) {
             return std::get<0>(pair) == std::get<1>(pair);
           });
  };
  std::function<bool(Value, OutputAxes)> canReplayEpilogue =
      [&](Value value, OutputAxes axes) {
    auto result = dyn_cast<FragmentType>(value.getType());
    if (value == reshape.getResult())
      return isSourceFrame(axes);
    if (!result)
      return true;
    if (axes.size() != result.getShape().size())
      return false;
    auto [entry, inserted] = checked.try_emplace(value, axes);
    if (!inserted) {
      bool changed = false;
      for (auto [known, current] : llvm::zip(entry->second, axes)) {
        if (known && current && known != current)
          return false;
        if (!known && current) {
          known = current;
          changed = true;
        }
      }
      if (!changed)
        return true;
      axes = entry->second;
    }
    Operation *producer = value.getDefiningOp();
    if (auto transpose = dyn_cast_or_null<TransposeOp>(producer)) {
      if (!transposes.contains(producer))
        return false;
      OutputAxes inputAxes(axes.size());
      auto relations = queryFragmentOperandRelations(transpose);
      if (failed(relations))
        return false;
      for (const FragmentAxisGroup &group : relations->front().groups)
        inputAxes[group.sourceAxes.front()] = axes[group.resultAxes.front()];
      return canReplayEpilogue(transpose.getValue(), std::move(inputAxes));
    }
    if (auto load = dyn_cast_or_null<LoadOp>(producer)) {
      if (!canReplayReadAt(load, store))
        return false;
      for (auto [axis, outputAxis] : llvm::enumerate(axes)) {
        PhysicalRangeFact ranges = analysis.axisRanges(value, axis);
        if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty())
          return false;
        if (ranges.roots.empty())
          continue;
        if (!outputAxis || *outputAxis >= outputRanges.size())
          return false;
        for (MakeRangeOp range : ranges.roots) {
          if (!analysis.lockstepRanges({range, outputRanges[*outputAxis]}).isExact())
            return false;
          auto [found, inserted] = companionAxes.try_emplace(range.getResult(), *outputAxis);
          if (!inserted && found->second != *outputAxis)
            return false;
        }
      }
      companions.push_back(load);
      return true;
    }
    if (auto companion = dyn_cast_or_null<ReshapeOp>(producer)) {
      auto source = cast<FragmentType>(companion.getValue().getType());
      auto primarySource = cast<FragmentType>(reshape.getValue().getType());
      auto primaryResult = cast<FragmentType>(reshape.getResult().getType());
      if (companion.getReassociation() == reshape.getReassociation() &&
          source.getShape() == primarySource.getShape() &&
          source.getAxisMaps() == primarySource.getAxisMaps() &&
          source.getOwner() == primarySource.getOwner() &&
          source.getValidity() == primarySource.getValidity() &&
          result.getShape() == primaryResult.getShape() &&
          result.getAxisMaps() == primaryResult.getAxisMaps() &&
          result.getOwner() == primaryResult.getOwner() &&
          result.getValidity() == primaryResult.getValidity()) {
        if (!isSourceFrame(axes))
          return false;
        if (!llvm::is_contained(sourceReshapes, companion))
          sourceReshapes.push_back(companion);
        return true;
      }
      auto relations = queryFragmentOperandRelations(companion);
      if (failed(relations))
        return false;
      OutputAxes inputAxes(source.getShape().size());
      for (const FragmentAxisGroup &group : relations->front().groups) {
        if (group.sourceAxes.empty()) {
          if (!unitAxes(result, group.resultAxes))
            return false;
        } else if (group.resultAxes.empty()) {
          if (!unitAxes(source, group.sourceAxes))
            return false;
        } else if (group.sourceAxes.size() != 1 || group.resultAxes.size() != 1) {
          return false;
        } else {
          inputAxes[group.sourceAxes.front()] = axes[group.resultAxes.front()];
        }
      }
      return canReplayEpilogue(companion.getValue(), std::move(inputAxes));
    }
    if (!isPointwise(producer))
      return false;
    return llvm::all_of(producer->getOperands(), [&](Value operand) {
      auto input = dyn_cast<FragmentType>(operand.getType());
      if (!input)
        return true;
      auto projection = queryBroadcastProjection(input, result);
      if (!projection.isExact())
        return false;
      OutputAxes inputAxes(input.getShape().size());
      for (auto [axis, inputAxis] : llvm::enumerate(projection.targetToSource)) {
        if (!inputAxis || !axes[axis])
          continue;
        if (inputAxes[*inputAxis] && inputAxes[*inputAxis] != axes[axis])
          return false;
        inputAxes[*inputAxis] = axes[axis];
      }
      return canReplayEpilogue(operand, std::move(inputAxes));
    });
  };
  OutputAxes outputAxes;
  for (unsigned axis = 0; axis < outputRank; ++axis)
    outputAxes.push_back(axis);
  if (!canReplayEpilogue(store.getValue(), std::move(outputAxes)))
    return false;
  SmallVector<Value> replayRoots{store.getValue()};
  if (store.getValid()) replayRoots.push_back(store.getValid());
  for (LoadOp companion : companions) {
    llvm::append_range(replayRoots, companion.getCoordinates());
    if (companion.getValid()) replayRoots.push_back(companion.getValid());
    if (companion.getFill()) replayRoots.push_back(companion.getFill());
  }
  // The input side of an aligned reshape is reused as existing SSA. Only its
  // output-frame epilogue and the accesses' actual coordinate suffixes replay.
  ReplayPolicy reuse(kernel, replayRoots, {store.getOperation()}, [&](Value value) {
    return llvm::none_of(sourceReshapes, [&](ReshapeOp source) {
      return source.getResult() == value;
    });
  });
  if (llvm::any_of(replayRoots, [&](Value root) {
        return reuse.duplicatesExpensiveWork(root);
      })) return false;
  auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
  if (!input)
    return false;
  auto relations = queryFragmentOperandRelations(reshape);
  if (failed(relations) ||
      cast<FragmentType>(relations->front().resultType).getShape().size() != outputRank)
    return false;
  const auto &groups = relations->front().groups;
  if (llvm::any_of(groups, [](const FragmentAxisGroup &group) {
        return group.kind != FragmentAxisRelationKind::Reassociation;
      }))
    return false;
  SmallVector<MakeRangeOp> inputRanges;
  SmallVector<Attribute> inputExtents;
  for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
    PhysicalRangeFact ranges = analysis.axisRanges(reshape.getValue(), axis);
    FailureOr<MakeRangeOp> range = queryExactLogicalRange(ranges);
    if (failed(range) && ranges.blockers.empty()) {
      auto traversal = analysis.lockstepRanges(ranges.roots);
      if (traversal.isExact())
        range = traversal.authority;
    }
    if (failed(range) || !isZero((*range).getStart()) ||
        !isZero((*range).getLogicalStart()) || !isUnitStepRange(*range))
      return false;
    auto realization = analysis.axisRealization(reshape.getValue(), axis);
    if (!realization.constructionScalarSeed &&
        !samePhysicalScalarExpression((*range).getExtent(), (*range).getLogicalStop()))
      return false;
    PhysicalExprAttr extent = queryLaunchExpression((*range).getLogicalStop());
    if (!extent)
      return false;
    inputRanges.push_back(*range);
    inputExtents.push_back(extent);
  }
  for (const FragmentAxisGroup &group : groups) {
    if (group.sourceAxes.empty() && group.resultAxes.empty())
      return false;
    SmallVector<Attribute> sourceExtents;
    for (unsigned axis : group.sourceAxes) {
      if (axis >= inputRanges.size())
        return false;
      sourceExtents.push_back(inputExtents[axis]);
    }
    SmallVector<Attribute> resultExtents;
    for (unsigned axis : group.resultAxes) {
      if (axis >= outerAxes.size())
        return false;
      PhysicalExprAttr extent = queryLaunchExpression(
          outputRanges[outerAxes[axis]].getLogicalStop());
      if (!extent)
        return false;
      resultExtents.push_back(extent);
    }
    // Check logical range extents, never provisional physical singletons.
    // The existing reassociation supplies the row-major axis order.
    if (failed(inferReshapeReassociation(
            store.getContext(), sourceExtents, resultExtents)))
      return false;
  }

  OpBuilder builder(store);
  SmallVector<Attribute> inputMappings;
  for (auto [axis, range] : llvm::enumerate(inputRanges)) {
    auto mapping = cast<AxisMapAttr>(range.getResult().getType().getAxisMaps()[0]);
    inputMappings.push_back(AxisMapAttr::get(
        store.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), axis, mapping.getDerived()));
  }
  input = FragmentType::get(
      store.getContext(), input.getElementType(), input.getShape(),
      builder.getArrayAttr(inputMappings), input.getValidity(), input.getOwner());
  auto indexType = FragmentType::get(
      input.getContext(), builder.getIndexType(), input.getShape(),
      input.getAxisMaps(), input.getValidity(), input.getOwner());
  SmallVector<Value> coordinates(store.getCoordinates());
  IRMapping mapping;
  for (const FragmentAxisGroup &group : groups) {
    Value ordinal;
    for (unsigned sourceAxis : group.sourceAxes) {
      auto rangeType = inputRanges[sourceAxis].getResult().getType();
      SmallVector<Attribute> axisShape(input.getShape().size(),
          PhysicalExprAttr::get(store.getContext(),
              PhysicalExprKind::Constant, 1,
              builder.getStringAttr(""), builder.getArrayAttr({})));
      axisShape[sourceAxis] = rangeType.getShape()[0];
      auto axisType = FragmentType::get(
          store.getContext(), builder.getIndexType(), builder.getArrayAttr(axisShape),
          input.getAxisMaps(), input.getValidity(), input.getOwner());
      SmallVector<Attribute> relation;
      for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
        SmallVector<int64_t> sourceAxes;
        if (axis == sourceAxis)
          sourceAxes.push_back(0);
        relation.push_back(ReshapeGroupAttr::get(
            store.getContext(), builder.getDenseI64ArrayAttr(sourceAxes),
            builder.getDenseI64ArrayAttr({static_cast<int64_t>(axis)})));
      }
      Value projected = builder.create<ReshapeOp>(
          store.getLoc(), axisType, inputRanges[sourceAxis].getResult(),
          builder.getArrayAttr(relation));
      if (axisType != indexType)
        projected = builder.create<BroadcastOp>(store.getLoc(), indexType, projected);
      if (ordinal) {
        Value extent = builder.create<SplatOp>(
            store.getLoc(), indexType, inputRanges[sourceAxis].getLogicalStop());
        ordinal = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, extent, BinaryOperator::Multiply);
        ordinal = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, projected, BinaryOperator::Add);
      } else {
        ordinal = projected;
      }
    }
    if (!ordinal) {
      Value zero = builder.create<arith::ConstantIndexOp>(store.getLoc(), 0);
      ordinal = builder.create<SplatOp>(store.getLoc(), indexType, zero);
    }
    ArrayRef<unsigned> axes = group.resultAxes;
    for (unsigned position = axes.size(); position-- > 0;) {
      unsigned outerAxis = outerAxes[axes[position]];
      MakeRangeOp range = outputRanges[outerAxis];
      Value coordinate = ordinal;
      if (position != 0) {
        Value one = builder.create<arith::ConstantIndexOp>(store.getLoc(), 1);
        // Empty logical domains have no active store.  A positive divisor
        // keeps their inactive physical lanes well-defined as well.
        Value divisor = builder.create<BinaryOp>(
            store.getLoc(), builder.getIndexType(), range.getLogicalStop(), one,
            BinaryOperator::Maximum);
        Value extent = builder.create<SplatOp>(store.getLoc(), indexType, divisor);
        coordinate = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, extent, BinaryOperator::Remainder);
        ordinal = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, extent, BinaryOperator::FloorDivide);
      }
      coordinates[coordinateSlots[outerAxis]] = coordinate;
      mapping.map(range.getResult(), coordinate);
      for (auto [root, axis] : companionAxes)
        if (axis == outerAxis)
          mapping.map(root, coordinate);
      kernel.walk([&](MakeRangeOp occurrence) {
        FailureOr<int64_t> occurrenceDimension = queryRangeDimension(occurrence);
        FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
        if (succeeded(occurrenceDimension) && succeeded(rangeDimension) &&
            *occurrenceDimension == *rangeDimension &&
            analysis.lockstepRanges({occurrence, range}).isExact())
          mapping.map(occurrence.getResult(), coordinate);
      });
    }
  }
  Value source = reshape.getValue();
  if (source.getType() != input)
    source = builder.create<BroadcastOp>(store.getLoc(), input, source);
  mapping.map(reshape.getResult(), source);
  for (ReshapeOp companion : llvm::drop_begin(sourceReshapes)) {
    auto original = cast<FragmentType>(companion.getValue().getType());
    auto target = FragmentType::get(
        input.getContext(), original.getElementType(), input.getShape(),
        input.getAxisMaps(), input.getValidity(), input.getOwner());
    FailureOr<Value> value = projectPhysicalValueToSchema(
        builder, store.getLoc(), companion.getValue(), target);
    if (failed(value))
      return store.emitOpError("aligned reshapes have no common input relation");
    mapping.map(companion.getResult(), *value);
  }
  FailureOr<Value> valid = replayFragmentValue(
      builder, store.getValid(), input, mapping, analysis, store);
  if (failed(valid))
    return store.emitOpError("flattened store could not preserve its access validity");
  auto predicate = FragmentType::get(
      input.getContext(), builder.getI1Type(), input.getShape(), input.getAxisMaps(),
      input.getValidity(), input.getOwner());
  Value zero = builder.create<arith::ConstantIndexOp>(store.getLoc(), 0);
  Value lower = builder.create<SplatOp>(store.getLoc(), indexType, zero);
  Value active = *valid;
  for (unsigned axis = 0; axis < outputRank; ++axis) {
    Value coordinate = coordinates[coordinateSlots[axis]];
    Value end = builder.create<SplatOp>(
        store.getLoc(), indexType, outputRanges[axis].getLogicalStop());
    Value nonNegative = builder.create<CompareOp>(
        store.getLoc(), predicate, coordinate, lower, ComparePredicate::Ge);
    Value belowEnd = builder.create<CompareOp>(
        store.getLoc(), predicate, coordinate, end, ComparePredicate::Lt);
    Value within = builder.create<BinaryOp>(
        store.getLoc(), predicate, nonNegative, belowEnd, BinaryOperator::LogicalAnd);
    active = active ? Value(builder.create<BinaryOp>(
        store.getLoc(), predicate, active, within, BinaryOperator::LogicalAnd)) : within;
  }
  for (LoadOp companion : companions) {
    auto companionType = FragmentType::get(
        input.getContext(), cast<FragmentType>(companion.getResult().getType()).getElementType(),
        input.getShape(), input.getAxisMaps(), input.getValidity(), input.getOwner());
    SmallVector<Value> companionCoordinates;
    for (Value coordinate : companion.getCoordinates()) {
      if (!isa<FragmentType>(coordinate.getType())) {
        companionCoordinates.push_back(coordinate);
        continue;
      }
      FailureOr<Value> projected = replayFragmentValue(
          builder, coordinate, indexType, mapping, analysis, store);
      if (failed(projected))
        return store.emitOpError(
            "reshaped store companion has no exact coordinate mapping")
               << "; coordinate=" << coordinate;
      companionCoordinates.push_back(*projected);
    }
    FailureOr<Value> companionValid = replayFragmentValue(
        builder, companion.getValid(), input, mapping, analysis, store);
    FailureOr<Value> companionFill = replayFragmentValue(
        builder, companion.getFill(), input, mapping, analysis, store);
    if (failed(companionValid) || failed(companionFill))
      return store.emitOpError(
          "reshaped store companion could not preserve validity and fill");
    companionValid = combinePredicates(
        builder, store.getLoc(), input, *companionValid, active);
    if (!*companionFill)
      companionFill = materializeZeroFragment(builder, store.getLoc(), companionType);
    if (failed(companionValid) || failed(companionFill))
      return failure();
    auto loaded = builder.create<LoadOp>(
        companion.getLoc(), companionType, companion.getResource(), companionCoordinates,
        *companionValid, *companionFill, companion.getSourceAxes());
    if (Attribute origin = companion->getAttr(originAttr))
      loaded->setAttr(originAttr, origin);
    mapping.map(companion.getResult(), loaded.getResult());
  }
  std::function<FailureOr<Value>(Value)> replayEpilogue = [&](Value value) -> FailureOr<Value> {
    if (Value mapped = mapping.lookupOrNull(value))
      return mapped;
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment)
      return value;
    Operation *producer = value.getDefiningOp();
    for (Value operand : producer->getOperands()) {
      FailureOr<Value> mapped = replayEpilogue(operand);
      if (failed(mapped))
        return failure();
      mapping.map(operand, *mapped);
    }
    auto target = FragmentType::get(
        input.getContext(), fragment.getElementType(), input.getShape(),
        input.getAxisMaps(), input.getValidity(), input.getOwner());
    if (isa<BroadcastOp, TransposeOp, ReshapeOp>(producer)) {
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, producer->getLoc(), mapping.lookup(producer->getOperand(0)), target);
      if (succeeded(projected))
        mapping.map(value, *projected);
      return projected;
    }
    auto cloned = cloneWithSchema(builder, producer, mapping, TypeRange{target});
    if (failed(cloned))
      return failure();
    return cloned->front();
  };
  FailureOr<Value> payload = replayEpilogue(store.getValue());
  if (failed(payload))
    return store.emitOpError("flattened store could not preserve its pointwise epilogue");
  auto replacement = builder.create<StoreOp>(
      store.getLoc(), store.getResource(), coordinates, *payload,
      active, store.getSourceAxes());
  if (Attribute origin = store->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  store.erase();
  return true;
}

} // namespace intent::gpu::access
