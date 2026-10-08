#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isZero(Value value) {
  PhysicalExprAttr expression = queryNonNegativeIndexUpperBound(value);
  return expression &&
         expression.getKind() ==
             PhysicalExprKind::Constant &&
         expression.getValue() == 0;
}

} // namespace

FailureOr<Value> materializeRetainedSlice(
    OpBuilder &builder, Location location, Value value, unsigned axis,
    PhysicalExprAttr blockedExtent, Value coordinates, Operation *insertionAnchor,
    AxisMapAttr resultMapping) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  auto original = dyn_cast<FragmentType>(value.getType());
  if (!kernel || !original || axis >= original.getShape().size() ||
      !DominanceInfo(kernel).dominates(value, insertionAnchor))
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  if (!analysis.axisRealization(value, axis).physicalized) {
    if (failed(realizeFullCoverageDimension(kernel, value, axis)))
      return kernel.emitError("retained value could not be fully materialized");
    original = cast<FragmentType>(value.getType());
    analysis = PhysicalProgramAnalysis(kernel);
  }
  PhysicalRangeFact ranges = analysis.axisRanges(value, axis);
  FailureOr<MakeRangeOp> authority = queryExactLogicalRange(ranges);
  if (failed(authority) || !isUnitStepRange(*authority) ||
      !analysis.lockstepRanges(ranges.roots).isExact() ||
      !analysis.axisRealization(value, axis).physicalized)
    return kernel.emitError("retained value has no realized slice coordinate relation");
  SmallVector<Attribute> shape(original.getShape().begin(), original.getShape().end());
  SmallVector<Attribute> mappings(original.getAxisMaps().begin(), original.getAxisMaps().end());
  shape[axis] = blockedExtent;
  if (resultMapping)
    mappings[axis] = AxisMapAttr::get(
        original.getContext(), resultMapping.getSourceId(),
        resultMapping.getSourceAxis(), resultMapping.getDimensionId(), axis,
        resultMapping.getDerived());
  auto target = FragmentType::get(original.getContext(), original.getElementType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
      original.getValidity(), original.getOwner());
  auto coordinate = cast<FragmentType>(coordinates.getType());
  Value start = builder.create<SplatOp>(location, coordinate, (*authority).getStart());
  Value ordinal = builder.create<BinaryOp>(location, coordinate, coordinates, start,
                                          BinaryOperator::Subtract);
  auto indexType = FragmentType::get(target.getContext(), builder.getIndexType(),
      target.getShape(), target.getAxisMaps(), target.getValidity(), target.getOwner());
  FailureOr<Value> indices = projectPhysicalValueToSchema(builder, location, ordinal, indexType);
  if (failed(indices))
    return kernel.emitError("retained slice has no index projection")
           << "; coordinate=" << coordinates << "; target=" << indexType
           << "; source=" << original;
  auto predicate = FragmentType::get(target.getContext(), builder.getI1Type(),
      target.getShape(), target.getAxisMaps(), target.getValidity(), target.getOwner());
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value lower = builder.create<SplatOp>(location, indexType, zero);
  Value extent = builder.create<SplatOp>(location, indexType, (*authority).getExtent());
  Value nonNegative = builder.create<CompareOp>(location, predicate, *indices, lower,
                                               ComparePredicate::Ge);
  Value inExtent = builder.create<CompareOp>(location, predicate, *indices, extent,
                                            ComparePredicate::Lt);
  Value valid = builder.create<BinaryOp>(location, predicate, nonNegative, inExtent,
                                         BinaryOperator::LogicalAnd);
  Value activeLength = builder.create<BinaryOp>(location, builder.getIndexType(),
      (*authority).getLogicalStop(), (*authority).getStart(), BinaryOperator::Subtract);
  Value activeExtent = builder.create<SplatOp>(location, indexType, activeLength);
  Value active = builder.create<CompareOp>(location, predicate, *indices, activeExtent,
                                          ComparePredicate::Lt);
  valid = builder.create<BinaryOp>(location, predicate, valid, active,
                                   BinaryOperator::LogicalAnd);
  FailureOr<Value> fill = materializeZeroFragment(builder, location, target);
  if (failed(fill))
    return failure();
  return Value(builder.create<GatherOp>(location, target, value, ValueRange{*indices},
      valid, *fill, ArrayRef<int64_t>{static_cast<int64_t>(axis)}));
}

namespace {

template <typename Emit>
LogicalResult buildStoreTraversal(OpBuilder &builder, Location location,
                                  ArrayRef<MakeRangeOp> ranges,
                                  ArrayRef<std::pair<MakeRangeOp, unsigned>> roots, Value rowChunk,
                                  FragmentType payload, unsigned chunkAxis, Emit emit) {
  OpBuilder::InsertionGuard guard(builder);
  IRMapping mapping;
  SmallVector<Value> coordinates;
  SmallVector<Value> predicates;
  for (unsigned axis = 0; axis < ranges.size(); ++axis) {
    MakeRangeOp range = ranges[axis];
    Value chunk = rowChunk;
    if (axis != chunkAxis)
      chunk = builder.create<PhysicalExprOp>(
          location, builder.getIndexType(),
          cast<PhysicalExprAttr>(payload.getShape()[axis]));
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value stop;
    if (auto constant = dyn_cast_or_null<IntegerAttr>(
            UniformValueAnalysis(describeUniformValue).evaluate(range.getLogicalStop())))
      stop = builder.create<arith::ConstantIndexOp>(location, constant.getInt());
    else
      stop = builder.create<PhysicalExprOp>(location, builder.getIndexType(),
                                           queryLaunchExpression(range.getLogicalStop()));
    Value start = isZero(range.getStart()) ? zero : range.getStart();
    if (axis == chunkAxis) {
      auto loop = builder.create<scf::ForOp>(location, zero, stop, chunk);
      builder.setInsertionPointToStart(loop.getBody());
      start = loop.getInductionVar();
    }
    auto original = range.getResult().getType();
    auto axisMap = cast<AxisMapAttr>(payload.getAxisMaps()[axis]);
    auto type = FragmentType::get(
        builder.getContext(), original.getElementType(),
        builder.getArrayAttr({queryLaunchExpression(chunk)}),
        builder.getArrayAttr({AxisMapAttr::get(builder.getContext(),
            axisMap.getSourceId(), axisMap.getSourceAxis(), axisMap.getDimensionId(),
            0, axisMap.getDerived())}), original.getValidity(), original.getOwner());
    auto current = builder.create<MakeRangeOp>(
        location, type, start, chunk, one, zero, stop, axisMap.getSourceId(),
        axisMap.getSourceAxis(), axisMap.getDerived());
    for (auto [root, rootAxis] : roots) {
      if (rootAxis != axis)
        continue;
      // A Cartesian value can use one range in more than one occurrence.
      // Producer replay binds the selected occurrence to the bounded traversal.
      if (axis != chunkAxis && llvm::any_of(roots, [&](auto entry) {
            return entry.first == root && entry.second == chunkAxis;
          }))
        continue;
      if (sameLogicalRange(root, range) &&
          root.getResult().getType().getAxisMaps() == type.getAxisMaps()) {
        mapping.map(root.getResult(), current.getResult());
        continue;
      }
      auto original = root.getResult().getType();
      auto rebound = FragmentType::get(
          builder.getContext(), original.getElementType(), type.getShape(),
          original.getAxisMaps(), original.getValidity(), original.getOwner());
      Value equivalent = builder.create<MakeRangeOp>(
          location, rebound, start, chunk, one, root.getLogicalStart(),
          root.getLogicalStop(), root.getSourceId(), root.getSourceAxis(), root.getDerived());
      mapping.map(root.getResult(), equivalent);
    }
    if (!mapping.contains(range.getResult()) &&
        (axis == chunkAxis || !llvm::any_of(roots, [&](auto entry) {
          return entry.first == range && entry.second == chunkAxis;
        })))
      mapping.map(range.getResult(), current.getResult());
    coordinates.push_back(current);
    auto boolean = FragmentType::get(
        builder.getContext(), builder.getI1Type(), type.getShape(),
        type.getAxisMaps(), type.getValidity(), type.getOwner());
    Value end = builder.create<BroadcastOp>(location, type, stop);
    predicates.push_back(builder.create<CompareOp>(
        location, boolean, current, end, ComparePredicate::Lt));
  }
  auto boolean = FragmentType::get(
      builder.getContext(), builder.getI1Type(), payload.getShape(),
      payload.getAxisMaps(), payload.getValidity(), payload.getOwner());
  Value valid;
  for (auto [axis, predicate] : llvm::enumerate(predicates)) {
    FailureOr<Value> projected =
        projectPredicateToFragmentAxis(builder, location, predicate, boolean, axis);
    if (failed(projected))
      return failure();
    valid = valid ? Value(builder.create<BinaryOp>(
                        location, boolean, valid, *projected,
                        BinaryOperator::LogicalAnd))
                  : *projected;
  }
  return emit(builder, mapping, coordinates, valid);
}

bool hasLaunchUniformBounds(scf::ForOp loop, func::FuncOp kernel) {
  return llvm::all_of(
      ValueRange{loop.getLowerBound(), loop.getUpperBound(), loop.getStep()},
      [&](Value bound) { return isLaunchUniformScalar(bound, kernel); });
}

bool materializeLoopState(scf::ForOp loop, func::FuncOp kernel) {
  if (loop->hasAttr(reductionSourcesAttr) ||
      !isProgramAllocationContext(loop, kernel))
    return false;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  for (auto [slot, initial] : llvm::enumerate(loop.getInitArgs())) {
    auto type = dyn_cast<FragmentType>(initial.getType());
    if (!type || type.getShape().empty() || type.getShape().size() > 2)
      continue;
    __int128 words = type.getElementType().isIndex()
                         ? 2
                         : std::max(1u, (type.getElementType().getIntOrFloatBitWidth() + 31) / 32);
    bool fixed = true;
    for (Attribute attribute : type.getShape()) {
      auto extent = cast<PhysicalExprAttr>(attribute);
      if (extent.getKind() != PhysicalExprKind::Constant ||
          extent.getValue() <= 0) {
        fixed = false;
        break;
      }
      if (words <= capabilities.getRegistersPerUnit())
        words *= extent.getValue();
    }
    if (!fixed || words <= capabilities.getRegistersPerUnit())
      continue;

    PhysicalProgramAnalysis analysis(kernel);
    DominanceInfo dominance(kernel);
    bool completeRanges = true;
    for (auto [axis, attribute] : llvm::enumerate(type.getShape())) {
      auto map = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
      auto facts = analysis.axisRanges(initial, axis);
      auto range = queryExactLogicalRange(facts);
      if (failed(range) || !analysis.lockstepRanges(facts.roots).isExact() ||
          !dominance.dominates((*range).getResult(), loop) ||
          !isZero((*range).getStart()) || !isZero((*range).getLogicalStart()) ||
          !isUnitStepRange(*range) ||
          queryLaunchExpression((*range).getExtent()) != attribute ||
          queryLaunchExpression((*range).getLogicalStop()) != attribute ||
          !(sourceAxisIdentity(*range) == sourceAxisIdentity(map))) {
        completeRanges = false;
        break;
      }
      auto dimension = queryRangeDimension(*range);
      if (failed(dimension) || *dimension != map.getDimensionId()) {
        completeRanges = false;
        break;
      }
      for (unsigned prior = 0; prior < axis; ++prior)
        if (sourceAxisIdentity(cast<AxisMapAttr>(type.getAxisMaps()[prior])) ==
            sourceAxisIdentity(map))
          completeRanges = false;
    }
    if (!completeRanges)
      continue;

    OpBuilder builder(loop);
    Value workspace = createProgramBuffer(
        builder, loop.getLoc(), type.getElementType(), type.getShape(),
        type.getOwner()).getResult();
    Value zero = builder.create<arith::ConstantIndexOp>(loop.getLoc(), 0);
    Value one = builder.create<arith::ConstantIndexOp>(loop.getLoc(), 1);
    SmallVector<Value> coordinates;
    SmallVector<int64_t> axes;
    auto predicateType = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), type.getShape(), type.getAxisMaps(),
        type.getValidity(), type.getOwner());
    Value valid;
    for (auto [axis, attribute] : llvm::enumerate(type.getShape())) {
      auto extent = cast<PhysicalExprAttr>(attribute);
      auto map = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
      auto coordinateType = FragmentType::get(
          kernel.getContext(), builder.getIndexType(), builder.getArrayAttr({extent}),
          builder.getArrayAttr({AxisMapAttr::get(
              kernel.getContext(), map.getSourceId(), map.getSourceAxis(),
              map.getDimensionId(), 0, map.getDerived())}),
          type.getValidity(), type.getOwner());
      Value end = builder.create<arith::ConstantIndexOp>(loop.getLoc(), extent.getValue());
      coordinates.push_back(builder.create<MakeRangeOp>(
          loop.getLoc(), coordinateType, zero, end, one, zero, end,
          map.getSourceId(), map.getSourceAxis(), map.getDerived()));
      axes.push_back(axis);
      auto axisPredicate = FragmentType::get(
          kernel.getContext(), builder.getI1Type(), coordinateType.getShape(),
          coordinateType.getAxisMaps(), type.getValidity(), type.getOwner());
      Value upper = builder.create<SplatOp>(loop.getLoc(), coordinateType, end);
      Value active = builder.create<CompareOp>(
          loop.getLoc(), axisPredicate, coordinates.back(), upper, ComparePredicate::Lt);
      active = builder.create<BroadcastOp>(loop.getLoc(), predicateType, active);
      valid = valid ? Value(builder.create<BinaryOp>(
                          loop.getLoc(), predicateType, valid, active,
                          BinaryOperator::LogicalAnd))
                    : active;
    }
    Value scalarFill = builder.create<arith::ConstantOp>(
        loop.getLoc(), builder.getZeroAttr(type.getElementType()));
    Value fill = builder.create<SplatOp>(loop.getLoc(), type, scalarFill);
    builder.create<StoreOp>(loop.getLoc(), workspace, coordinates, initial, valid, axes);

    SmallVector<Value> remaining;
    for (auto [position, value] : llvm::enumerate(loop.getInitArgs()))
      if (position != slot)
        remaining.push_back(value);
    auto replacement = builder.create<scf::ForOp>(
        loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(), remaining);
    replacement->setAttrs(loop->getAttrs());
    if (!replacement.getBody()->empty())
      replacement.getBody()->back().erase();
    builder.setInsertionPointToStart(replacement.getBody());
    Value current = builder.create<LoadOp>(
        loop.getLoc(), type, workspace, coordinates, valid, fill, axes);
    loop.getInductionVar().replaceAllUsesWith(replacement.getInductionVar());
    unsigned next = 0;
    for (auto [position, argument] : llvm::enumerate(loop.getRegionIterArgs()))
      argument.replaceAllUsesWith(position == slot
                                      ? current
                                      : replacement.getRegionIterArgs()[next++]);
    replacement.getBody()->getOperations().splice(
        replacement.getBody()->end(), loop.getBody()->getOperations());
    auto yield = cast<scf::YieldOp>(replacement.getBody()->getTerminator());
    builder.setInsertionPoint(yield);
    builder.create<StoreOp>(loop.getLoc(), workspace, coordinates,
                            yield.getOperand(slot), valid, axes);
    remaining.clear();
    for (auto [position, value] : llvm::enumerate(yield.getOperands()))
      if (position != slot)
        remaining.push_back(value);
    builder.create<scf::YieldOp>(loop.getLoc(), remaining);
    yield.erase();

    builder.setInsertionPointAfter(replacement);
    Value final = builder.create<LoadOp>(
        loop.getLoc(), type, workspace, coordinates, valid, fill, axes);
    next = 0;
    for (auto [position, result] : llvm::enumerate(loop.getResults()))
      result.replaceAllUsesWith(position == slot ? final : replacement.getResult(next++));
    loop.erase();
    return true;
  }
  return false;
}

FailureOr<bool> materializeRetainedStoreAlongAxis(StoreOp store,
                                                func::FuncOp kernel,
                                                unsigned chunkAxis) {
  auto payload = dyn_cast<FragmentType>(store.getValue().getType());
  auto output = dyn_cast<ViewType>(store.getResource().getType());
  auto buffer = dyn_cast<BufferType>(store.getResource().getType());
  if (!output && !buffer)
    return false;
  unsigned rank = output ? output.getRank() : buffer.getShape().size();
  ArrayAttr outputShape = output ? output.getLayout().getExtents() : buffer.getShape();
  if (!payload ||
      !((payload.getShape().size() == 2 && rank == 2 &&
         store.getSourceAxes() == ArrayRef<int64_t>{0, 1}) ||
        (payload.getShape().size() == 1 && rank == 1 &&
         store.getSourceAxes() == ArrayRef<int64_t>{0})))
    return false;
  auto enclosingLoop = store->getParentOfType<scf::ForOp>();
  for (Operation *parent = store->getParentOp(); parent != kernel;
       parent = parent->getParentOp()) {
    if (auto group = dyn_cast<ExecutionGroupOp>(parent);
        group && isSingletonExecutionGroup(group, kernel)) {
      continue;
    } else if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      if (!hasLaunchUniformBounds(loop, kernel))
        return false;
    } else if (auto branch = dyn_cast<scf::IfOp>(parent)) {
      if (enclosingLoop && !isLaunchUniformScalar(branch.getCondition(), kernel))
        return false;
    } else {
      return false;
    }
  }
  PhysicalProgramAnalysis analysis(kernel);
  bool indirect = payload.getShape().size() == 1 &&
                  !store.getCoordinates().front().getDefiningOp<MakeRangeOp>();
  if ((indirect || buffer) ? !analysis.accessBounds(store).isExact()
                           : !analysis.boundaryValidity(store).isExact())
    return false;
  SmallVector<Value> retained{store.getValue()};
  if (indirect) {
    retained.push_back(store.getCoordinates().front());
    if (store.getValid())
      retained.push_back(store.getValid());
    if (!llvm::all_of(retained, [&](Value value) {
          auto type = dyn_cast<FragmentType>(value.getType());
          return type && type.getShape().size() == 1 &&
                 type.getShape() == payload.getShape();
        }))
      return false;
  }
  SmallVector<MakeRangeOp> ranges;
  SmallVector<Attribute> shape;
  for (auto [axis, coordinate] : llvm::enumerate(store.getCoordinates())) {
    auto range = coordinate.getDefiningOp<MakeRangeOp>();
    if (indirect) {
      auto facts = analysis.axisRanges(store.getValue(), axis);
      if (facts.state == PhysicalFactState::Unknown || facts.roots.empty() || !facts.blockers.empty() ||
          !analysis.lockstepRanges(facts.roots).isExact())
        return false;
      range = facts.roots.front();
    }
    auto extent = cast<PhysicalExprAttr>(payload.getShape()[axis]);
    if (!range || !isZero(range.getStart()) || !isZero(range.getLogicalStart()) ||
        !isUnitStepRange(range))
      return false;
    PhysicalExprAttr end = queryLaunchExpression(range.getLogicalStop());
    if (!end || (!indirect && end != outputShape[axis]))
      return false;
    if (extent.getKind() == PhysicalExprKind::Parameter) {
      FailureOr<ParameterAttr> parameter = queryParameterBySymbol(kernel, extent.getParameterReference().getName());
      if (failed(parameter) || !parameter->isDeferred())
        return false;
    } else if (extent.getKind() != PhysicalExprKind::Constant ||
               extent != end) {
      return false;
    }
    ranges.push_back(range);
    shape.push_back(end);
  }
  if (buffer && !indirect) {
    SmallVector<std::pair<MakeRangeOp, Value>> boundaries;
    for (MakeRangeOp range : ranges)
      boundaries.emplace_back(range, range.getLogicalStop());
    if (!analysis.isTailPredicate(store.getValid(), boundaries))
      return false;
  }
  SmallVector<Value> dependencies(retained);
  llvm::DenseSet<Operation *> visited;
  for (unsigned index = 0; index < dependencies.size(); ++index) {
    Operation *producer = dependencies[index].getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (isa<ScanOp>(producer))
      return false;
    dependencies.append(producer->getOperands().begin(),
                        producer->getOperands().end());
  }
  SmallVector<MakeRangeOp> roots;
  for (Value value : retained) {
    PhysicalRangeFact facts = analysis.sourceRanges(value);
    for (MakeRangeOp root : facts.roots)
      if (!llvm::is_contained(roots, root))
        roots.push_back(root);
  }
  roots.append(ranges.begin(), ranges.end());
  SmallVector<std::pair<MakeRangeOp, unsigned>> rootBindings;
  for (MakeRangeOp root : roots) {
    std::optional<unsigned> selected;
    bool exactSource = false;
    for (auto [axis, range] : llvm::enumerate(ranges)) {
      auto rootDimension = queryRangeDimension(root);
      auto rangeDimension = queryRangeDimension(range);
      if (!isZero(root.getStart()) || failed(rootDimension) ||
          failed(rangeDimension) || *rootDimension != *rangeDimension ||
          !analysis.lockstepRanges({root, range}).isExact())
        continue;
      bool sameSource = sameLogicalRange(root, range);
      if (selected && sameSource == exactSource)
        return false;
      if (!selected || sameSource) {
        selected = axis;
        exactSource = sameSource;
      }
    }
    if (!selected)
      return false;
    rootBindings.emplace_back(root, *selected);
  }
  SmallVector<MakeRangeOp> chunkRoots;
  for (auto [root, axis] : rootBindings)
    if (axis == chunkAxis && !llvm::is_contained(chunkRoots, root))
      chunkRoots.push_back(root);
  bool independent = llvm::all_of(visited, [&](Operation *producer) {
    auto reduce = dyn_cast<ReduceOp>(producer);
    if (!reduce)
      return true;
    return llvm::all_of(
        reduce.getSources(), [&](Value input) {
          auto axes = analysis.rangeAxes(input, chunkRoots);
          return axes.isExact() && llvm::none_of(axes.fragmentAxes, [&](unsigned axis) {
            return llvm::is_contained(reduce.getAxes(), axis);
          });
        });
  });
  if (!independent)
    return false;
  PhysicalSourceAxis source = sourceAxisIdentity(ranges[chunkAxis]);
  SmallVector<std::pair<PhysicalSourceAxis, int64_t>> replayAxes;
  for (Value value : retained) {
    auto mapping = cast<AxisMapAttr>(
        cast<FragmentType>(value.getType()).getAxisMaps()[chunkAxis]);
    std::pair axis{sourceAxisIdentity(mapping), mapping.getDimensionId()};
    if (!llvm::is_contained(replayAxes, axis))
      replayAxes.push_back(axis);
    auto facts = analysis.axisRanges(value, chunkAxis);
    for (MakeRangeOp root : facts.roots) {
      auto mapping = cast<AxisMapAttr>(root.getResult().getType().getAxisMaps()[0]);
      std::pair axis{sourceAxisIdentity(root), mapping.getDimensionId()};
      if (!llvm::is_contained(replayAxes, axis))
        replayAxes.push_back(axis);
    }
  }
  IRMapping replayBindings;
  auto replayAt = [&](Operation *anchor, bool &blocked, bool allowContractions) {
    bool reads = false;
    for (Value value : retained)
      for (auto [axis, dimension] : replayAxes) {
        auto replay = analysis.replayAt(value, axis,
            PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true, anchor,
            replayBindings, dimension);
        if (!replay.isReplayable())
          return false;
        if (!allowContractions && !replay.contractions.empty())
          return false;
        for (Operation *access : replay.accesses) {
          auto load = dyn_cast<LoadOp>(access);
          if (!load)
            return false;
          reads = true;
        }
        blocked |= !analysis.replayAt(
            value, axis, PhysicalReplayScope::ValueGraph,
            /*allowAccesses=*/true, store, replayBindings, dimension).isReplayable();
      }
    return reads || static_cast<bool>(buffer);
  };
  // Preserve the read snapshot before the first clobber, then perform the
  // external write at its original position using the private saved value.
  StoreOp clobber;
  for (Operation &operation : store->getBlock()->without_terminator()) {
    if (&operation == store)
      break;
    auto write = dyn_cast<StoreOp>(operation);
    if (!write || !isa<ViewType>(write.getResource().getType()))
      continue;
    bool blocked = false;
    bool available = replayAt(write, blocked, /*allowContractions=*/true);
    if (available && blocked) {
      clobber = write;
      break;
    }
  }
  if (!clobber) {
    auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
    __int128 footprint = std::max(1u,
        (payload.getElementType().getIntOrFloatBitWidth() + 31) / 32);
    bool fixed = true;
    for (Attribute attribute : shape) {
      auto extent = cast<PhysicalExprAttr>(attribute);
      fixed &= extent.getKind() == PhysicalExprKind::Constant;
      if (fixed)
        footprint *= extent.getValue();
    }
    // A launch-dependent full tensor has no proven register bound. Use the
    // same bounded traversal as an oversized static tensor; its workspace
    // still follows the original runtime shape and store order.
    // The payload must leave registers for addresses and traversal state.
    if (!fixed || footprint >= capabilities.getRegistersPerUnit()) {
      bool blocked = false;
      // Contraction blocking owns its result footprint and program slices.
      if (replayAt(store, blocked, /*allowContractions=*/false))
        clobber = store;
    }
  }
  if (!clobber)
    return false;
  DominanceInfo dominance(kernel);
  SmallVector<std::pair<Value, TypedAttr>> constants;
  UniformValueAnalysis uniform(describeUniformValue);
  for (Value dependency : dependencies) {
    if (dominance.dominates(dependency, clobber))
      continue;
    if (!isa<FragmentType>(dependency.getType())) {
      auto constant = dyn_cast_or_null<TypedAttr>(uniform.evaluate(dependency));
      if (!constant || constant.getType() != dependency.getType())
        return false;
      constants.emplace_back(dependency, constant);
      continue;
    }
    PhysicalRangeAxisFact axes = analysis.rangeAxes(dependency, chunkRoots);
    // The mapped ranges also reconstruct pure column coordinates and their
    // predicates. These do not carry a row axis; data reads still require it.
    if (axes.isExact() && axes.fragmentAxes.empty() &&
        analysis.replayability(dependency, std::nullopt,
                               PhysicalReplayScope::Coordinate,
                               /*allowAccesses=*/false).isReplayable())
      continue;
    bool scalarBroadcast = dependency.getDefiningOp<SplatOp>() != nullptr;
    if (auto broadcast = dependency.getDefiningOp<BroadcastOp>())
      scalarBroadcast |=
          !isa<FragmentType, RecordType>(broadcast.getValue().getType());
    bool projectedBroadcast = scalarBroadcast && llvm::any_of(
        chunkRoots, [&](MakeRangeOp root) {
          return queryFragmentAxis(dependency.getType(), sourceAxisIdentity(root))
              .isExact();
        });
    if (!projectedBroadcast &&
        (!axes.isExact() || axes.fragmentAxes.size() != 1))
      return false;
  }

  auto preservesUnreadChunks = [&] {
    if (output) {
      SmallVector<std::pair<MakeRangeOp, Value>> boundaries;
      for (MakeRangeOp range : ranges)
        boundaries.emplace_back(range, range.getLogicalStop());
      // A data-dependent write mask can itself observe the destination. Only
      // range validity may be reconstructed while streaming a direct output.
      if (!analysis.isTailPredicate(store.getValid(), boundaries))
        return false;
    }
    for (Value value : retained)
      for (auto [axis, dimension] : replayAxes) {
        auto replay = analysis.replayAt(
            value, axis, PhysicalReplayScope::ValueGraph,
            /*allowAccesses=*/true, store, replayBindings, dimension);
        if (!replay.isReplayable())
          return false;
        for (Operation *access : replay.accesses) {
          auto load = dyn_cast<LoadOp>(access);
          if (!load)
            return false;
          if (load.getResource() != store.getResource()) {
            if (output) {
              if (isa<ViewType>(load.getResource().getType())) {
                auto input = getPublicView(load.getResource());
                auto destination = getPublicView(store.getResource());
                bool independent = isInvocationWorkspace(load.getResource()) ||
                    isInvocationWorkspace(store.getResource()) ||
                    (input && input.getConstraints().getNoalias()) ||
                    (destination && destination.getConstraints().getNoalias());
                if (!independent) return false;
              }
            }
            continue;
          }
          if (output)
            return false;
          auto position = llvm::find(load.getSourceAxes(), chunkAxis);
          if (position == load.getSourceAxes().end())
            return false;
          auto range = load.getCoordinates()[position - load.getSourceAxes().begin()]
                           .getDefiningOp<MakeRangeOp>();
          // Replayed destination reads must stay inside the block written by
          // this traversal. Invariant SSA captures retain their old values.
          if (!range || !llvm::is_contained(chunkRoots, range))
            return false;
        }
      }
    return true;
  };
  bool direct = !indirect && clobber == store && preservesUnreadChunks();
  OpBuilder builder(clobber);
  IRMapping constantValues;
  for (auto [original, constant] : constants)
    constantValues.map(original, builder.create<arith::ConstantOp>(
        store.getLoc(), original.getType(), constant));
  SmallVector<Value> workspaces;
  for (Value value : retained) {
    auto type = cast<FragmentType>(value.getType());
    workspaces.push_back(direct ? store.getResource() :
        createInvocationBuffer(kernel, store.getLoc(),
            type.getElementType(), builder.getArrayAttr(shape),
            type.getOwner()).getResult());
  }
  Value workspace = workspaces.front();
  auto workspaceBuffer = dyn_cast<BufferType>(workspace.getType());
  std::string resourceId = workspaceBuffer
      ? std::to_string(workspaceBuffer.getInstance())
      : ("VIEW_" + Twine(cast<ViewType>(workspace.getType()).getSourceId())).str();
  bool linearTraversal = payload.getShape().size() == 1;
  auto reference = getOrCreatePhysicalParameter(
      kernel, ((linearTraversal ? "MATERIALIZE_ELEMENTS_" : "MATERIALIZE_AXIS_") +
               resourceId),
      linearTraversal ? ParameterRole::OwnershipN : ParameterRole::ReductionOuter,
      linearTraversal ? ParameterCategory::Pointwise : ParameterCategory::Reduction,
      payload.getElementType().getIntOrFloatBitWidth(),
      linearTraversal
          ? ArrayRef<int64_t>{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}
          : ArrayRef<int64_t>{1, 2, 4, 8, 16, 32, 64});
  if (failed(reference)) return failure();
  OpBuilder entry(&kernel.front(), kernel.front().begin());
  auto chunk = materializeParameter(entry, store.getLoc(), *reference);
  SmallVector<Attribute> blockedShape(payload.getShape().begin(),
                                      payload.getShape().end());
  blockedShape[chunkAxis] = queryLaunchExpression(chunk);
  auto blocked = FragmentType::get(
      kernel.getContext(), payload.getElementType(),
      builder.getArrayAttr(blockedShape), payload.getAxisMaps(),
      payload.getValidity(), payload.getOwner());
  ReplayMaterializationOptions options;
  options.fragmentAxis = chunkAxis;
  options.traversalRanges = chunkRoots;
  LogicalResult initialized = buildStoreTraversal(
      builder, store.getLoc(), ranges, rootBindings, chunk, blocked, chunkAxis,
      [&](OpBuilder &nested, IRMapping &mapping, ValueRange coordinates,
          Value valid) -> LogicalResult {
        for (const auto &entry : constantValues.getValueMap())
          mapping.map(entry.first, entry.second);
        if (direct && store.getValid()) {
          auto predicate = materializeReplayedValue(
              nested, store.getLoc(), store.getValid(), source,
              queryLaunchExpression(chunk), mapping, clobber, options);
          if (failed(predicate))
            return store.emitOpError("buffer write validity cannot be replayed");
          auto combined = materializeValidityConjunction(
              nested, store.getLoc(), valid, *predicate, blocked);
          if (failed(combined))
            return failure();
          valid = *combined;
        }
        for (auto [original, snapshot] : llvm::zip(retained, workspaces)) {
          FailureOr<Value> value = materializeReplayedValue(
              nested, store.getLoc(), original, source,
              queryLaunchExpression(chunk), mapping, clobber, options);
          if (failed(value))
            return store.emitOpError("retained value has no bounded producer replay");
          auto written = nested.create<StoreOp>(store.getLoc(), snapshot,
              coordinates, *value, valid, store.getSourceAxes());
          if (direct)
            written->setAttrs(store->getAttrs());
        }
        return success();
      });
  if (failed(initialized))
    return failure();
  if (direct) {
    store.erase();
    eraseDeadPhysicalValues(kernel);
    return true;
  }
  builder.setInsertionPoint(store);
  LogicalResult copied = buildStoreTraversal(
      builder, store.getLoc(), ranges, rootBindings, chunk, blocked, chunkAxis,
      [&](OpBuilder &nested, IRMapping &mapping, ValueRange coordinates,
          Value valid) -> LogicalResult {
        if (store.getValid() && !indirect) {
          FailureOr<Value> predicate = materializeReplayedValue(
              nested, store.getLoc(), store.getValid(), source,
              queryLaunchExpression(chunk), mapping, store, options);
          if (failed(predicate))
            return store.emitOpError("retained output validity cannot be replayed");
          auto type = cast<FragmentType>(valid.getType());
          valid = nested.create<BinaryOp>(store.getLoc(), type, valid, *predicate,
                                          BinaryOperator::LogicalAnd);
        }
        SmallVector<Value> restored;
        for (auto [original, snapshot] : llvm::zip(retained, workspaces)) {
          auto type = cast<FragmentType>(original.getType());
          auto current = FragmentType::get(kernel.getContext(), type.getElementType(),
              blocked.getShape(), type.getAxisMaps(), type.getValidity(), type.getOwner());
          auto zero = materializeZeroFragment(nested, store.getLoc(), current);
          if (failed(zero))
            return failure();
          restored.push_back(nested.create<LoadOp>(store.getLoc(), current, snapshot,
              coordinates, valid, *zero, store.getSourceAxes()));
        }
        SmallVector<Value> destinations(coordinates);
        if (indirect) {
          destinations[0] = restored[1];
          if (store.getValid()) {
            auto predicate = projectPhysicalValueToSchema(nested, store.getLoc(),
                restored[2], cast<FragmentType>(valid.getType()));
            if (failed(predicate))
              return failure();
            valid = nested.create<BinaryOp>(store.getLoc(), valid.getType(), valid,
                                            *predicate, BinaryOperator::LogicalAnd);
          }
          // The original access was in bounds. Retain that executable proof
          // after its address and predicate have become separate saved values.
          auto originalIndex = cast<FragmentType>(destinations[0].getType());
          auto indexType = FragmentType::get(kernel.getContext(), nested.getIndexType(),
              originalIndex.getShape(), originalIndex.getAxisMaps(),
              originalIndex.getValidity(), originalIndex.getOwner());
          if (originalIndex != indexType)
            destinations[0] = nested.create<CastOp>(store.getLoc(), indexType, destinations[0]);
          Value zero = nested.create<arith::ConstantIndexOp>(store.getLoc(), 0);
          Value end = nested.create<DimOp>(store.getLoc(), nested.getIndexType(),
                                          store.getResource(), store.getSourceAxes()[0]);
          for (auto [bound, predicate] :
               {std::pair{zero, ComparePredicate::Ge}, std::pair{end, ComparePredicate::Lt}}) {
            Value limit = nested.create<BroadcastOp>(store.getLoc(), indexType, bound);
            Value inBounds = nested.create<CompareOp>(store.getLoc(), valid.getType(),
                                                      destinations[0], limit, predicate);
            valid = nested.create<BinaryOp>(store.getLoc(), valid.getType(), valid,
                                            inBounds, BinaryOperator::LogicalAnd);
          }
        }
        auto replacement = nested.create<StoreOp>(
            store.getLoc(), store.getResource(), destinations, restored.front(), valid,
            store.getSourceAxes());
        replacement->setAttrs(store->getAttrs());
        return success();
      });
  if (failed(copied))
    return failure();
  store.erase();
  eraseDeadPhysicalValues(kernel);
  return true;
}

FailureOr<bool> materializeRetainedStore(StoreOp store, func::FuncOp kernel) {
  auto payload = dyn_cast<FragmentType>(store.getValue().getType());
  if (!payload || payload.getShape().empty() || payload.getShape().size() > 2)
    return false;
  for (unsigned axis = 0; axis < payload.getShape().size(); ++axis) {
    auto result = materializeRetainedStoreAlongAxis(store, kernel, axis);
    if (failed(result) || *result)
      return result;
  }
  return false;
}

FailureOr<bool> materializeRetainedGather(GatherOp gather, func::FuncOp kernel) {
  Value source = gather.getSource();
  auto payload = dyn_cast<FragmentType>(source.getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!payload || payload.getShape().empty() ||
      gather.getCoordinates().size() != payload.getShape().size() ||
      !isa<FloatType, IntegerType>(payload.getElementType()))
    return false;
  unsigned rank = payload.getShape().size();
  auto coverageBound = [&](unsigned axis) -> PhysicalExprAttr {
    auto extent = cast<PhysicalExprAttr>(payload.getShape()[axis]);
    if (extent.getKind() != PhysicalExprKind::Parameter)
      return {};
    auto parameter = queryParameterBySymbol(kernel, extent.getParameterReference().getName());
    if (failed(parameter))
      return {};
    auto dimension = parameter->isDeferred() ? parameter->getBinding().getDimension() : IntegerAttr();
    auto mapping = cast<AxisMapAttr>(payload.getAxisMaps()[axis]);
    if (!dimension || dimension.getInt() != mapping.getDimensionId())
      return {};
    return parameter->getBinding().getCoverageBound();
  };
  unsigned chunkAxis = 0;
  int64_t largest = 0;
  bool dynamicCoverage = false;
  for (auto [axis, attribute] : llvm::enumerate(payload.getShape())) {
    auto bound = coverageBound(axis);
    if (bound && bound.getKind() == PhysicalExprKind::Dimension) {
      chunkAxis = axis;
      dynamicCoverage = true;
      break;
    }
    auto extent = cast<PhysicalExprAttr>(attribute);
    if (extent.getKind() == PhysicalExprKind::Constant &&
        extent.getValue() > largest) {
      largest = extent.getValue();
      chunkAxis = axis;
    }
  }
  auto full = cast<PhysicalExprAttr>(payload.getShape()[chunkAxis]);
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!dynamicCoverage &&
      full.getKind() != PhysicalExprKind::Constant)
    return false;
  auto footprint = minimumFragmentRegisterFootprint(kernel, source,
      capabilities.getRegistersPerUnit(), FragmentFootprintScope::PhysicalShape);
  if (!footprint || (!dynamicCoverage && *footprint <= capabilities.getRegistersPerUnit()))
    return false;
  Operation *definition = source.getDefiningOp();
  if (!definition)
    return false;
  SmallVector<GatherOp> readers;
  for (Operation *user : source.getUsers()) {
    auto reader = dyn_cast<GatherOp>(user);
    if (!reader || reader.getSource() != source ||
        reader.getCoordinates().size() != rank)
      return false;
    readers.push_back(reader);
  }
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<MakeRangeOp> ranges;
  SmallVector<std::pair<MakeRangeOp, unsigned>> rootBindings;
  SmallVector<MakeRangeOp> traversalRoots;
  SmallVector<Attribute> shape;
  SmallVector<unsigned> ownedAxes;
  for (unsigned axis = 0; axis < rank; ++axis) {
    auto facts = analysis.axisRanges(source, axis);
    if (facts.roots.empty() || !facts.blockers.empty() ||
        !analysis.lockstepRanges(facts.roots).isExact())
      return false;
    MakeRangeOp range = facts.roots.front();
    if (!isZero(range.getLogicalStart()) || !isUnitStepRange(range))
      return false;
    auto extent = queryLaunchExpression(range.getLogicalStop());
    if (auto constant = dyn_cast_or_null<IntegerAttr>(
            UniformValueAnalysis(describeUniformValue).evaluate(range.getLogicalStop())))
      extent = PhysicalExprAttr::get(kernel.getContext(),
          PhysicalExprKind::Constant, constant.getInt(),
          StringAttr::get(kernel.getContext(), ""), ArrayAttr::get(kernel.getContext(), {}));
    if (!extent ||
        (extent.getKind() != PhysicalExprKind::Constant &&
         extent.getKind() != PhysicalExprKind::Dimension))
      return false;
    if (analysis.isProgramOwnedRange(range))
      ownedAxes.push_back(axis);
    else if (!isZero(range.getStart()) ||
             (extent != payload.getShape()[axis] && extent != coverageBound(axis)))
      return false;
    ranges.push_back(range);
    shape.push_back(extent);
    for (MakeRangeOp root : facts.roots) {
      rootBindings.emplace_back(root, axis);
      if (axis == chunkAxis)
        traversalRoots.push_back(root);
    }
  }
  if (llvm::any_of(ownedAxes, [&](unsigned axis) { return axis >= chunkAxis; }))
    return false;
  if (!result && !ownedAxes.empty())
    return false;
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (ownedAxes.empty() && !llvm::all_of(space, [](Attribute attribute) {
        auto extent = cast<PhysicalExprAttr>(attribute);
        return extent.getKind() == PhysicalExprKind::Constant &&
               extent.getValue() == 1;
      }))
    return false;
  auto ordinalRange = [](Value value) {
    while (value) {
      if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
        value = broadcast.getValue();
        continue;
      }
      if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
        auto relations = queryFragmentOperandRelations(reshape.getOperation());
        if (failed(relations) ||
            llvm::any_of(relations->front().groups, [](const auto &group) {
              return group.sourceAxes.size() > 1 || group.resultAxes.size() > 1;
            }))
          return MakeRangeOp();
        value = reshape.getValue();
        continue;
      }
      break;
    }
    return value.getDefiningOp<MakeRangeOp>();
  };
  for (GatherOp reader : readers)
    for (unsigned axis : ownedAxes) {
      auto position = llvm::find(reader.getSourceAxes(), axis);
      if (position == reader.getSourceAxes().end())
        return false;
      auto ordinal = ordinalRange(reader.getCoordinates()[position - reader.getSourceAxes().begin()]);
      if (!ordinal || !isZero(ordinal.getStart()) ||
          !isZero(ordinal.getLogicalStart()) || !isUnitStepRange(ordinal) ||
          queryLaunchExpression(ordinal.getExtent()) != queryLaunchExpression(ranges[axis].getExtent()) ||
          queryLaunchExpression(ordinal.getLogicalStop()) != queryLaunchExpression(ranges[axis].getExtent()))
        return false;
    }
  Operation *insertionAnchor = definition->getNextNode();
  if (!insertionAnchor)
    return false;
  PhysicalSourceAxis axis = sourceAxisIdentity(ranges[chunkAxis]);
  IRMapping replayBindings;
  auto replay = analysis.replayAt(source, axis, PhysicalReplayScope::ValueGraph,
                                 /*allowAccesses=*/true, insertionAnchor,
                                 replayBindings);
  if (!replay.isReplayable() || llvm::any_of(replay.accesses, [&](Operation *access) {
        return !isa<LoadOp>(access);
      }))
    return false;
  SmallVector<Value> dependencies{source};
  llvm::DenseSet<Operation *> visited;
  for (unsigned index = 0; index < dependencies.size(); ++index) {
    Operation *producer = dependencies[index].getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (producer->getNumRegions() || isa<GatherOp>(producer))
      return false;
    dependencies.append(producer->getOperands().begin(), producer->getOperands().end());
  }
  OpBuilder builder(insertionAnchor);
  PhysicalExprAttr chunkExtent;
  Value chunk;
  bool readerChunk = rank == 1 && result && result.getShape().size() == 1;
  if (readerChunk) {
    chunkExtent = cast<PhysicalExprAttr>(result.getShape()[0]);
    if (chunkExtent == full)
      return false;
  }
  Value workspace = createInvocationBuffer(
      kernel, gather.getLoc(), payload.getElementType(),
      builder.getArrayAttr(shape), payload.getOwner()).getResult();
  if (!readerChunk) {
    auto instance = cast<BufferType>(workspace.getType()).getInstance();
    auto parameter = getOrCreatePhysicalParameter(kernel,
        ("MATERIALIZE_AXIS_" + Twine(instance)).str(),
        ParameterRole::ReductionOuter, ParameterCategory::Reduction,
        payload.getElementType().getIntOrFloatBitWidth(),
        {64, 128, 256, 512, 1024, 2048});
    if (failed(parameter)) return failure();
    chunkExtent = PhysicalExprAttr::get(kernel.getContext(),
        PhysicalExprKind::Parameter, 0, *parameter,
        builder.getArrayAttr({}));
  }
  if (chunkExtent.getKind() == PhysicalExprKind::Parameter) {
    FailureOr<ParameterAttr> parameter = queryParameterBySymbol(kernel, chunkExtent.getParameterReference().getName());
    if (failed(parameter))
      return failure();
    chunk = materializeParameter(builder, gather.getLoc(), parameter->getReference());
  } else {
    chunk = builder.create<arith::ConstantIndexOp>(gather.getLoc(), chunkExtent.getValue());
  }
  SmallVector<Attribute> blockedShape(payload.getShape().begin(), payload.getShape().end());
  blockedShape[chunkAxis] = chunkExtent;
  auto blocked = FragmentType::get(kernel.getContext(), payload.getElementType(),
      builder.getArrayAttr(blockedShape), payload.getAxisMaps(),
      payload.getValidity(), payload.getOwner());
  ReplayMaterializationOptions options;
  options.fragmentAxis = chunkAxis;
  options.traversalRanges = traversalRoots;
  options.materializeZeroFill = true;
  SmallVector<int64_t> sourceAxes;
  for (unsigned axis = 0; axis < rank; ++axis)
    sourceAxes.push_back(axis);
  SmallVector<Value> ownedCoordinates(rank);
  if (failed(buildStoreTraversal(
          builder, gather.getLoc(), ranges, rootBindings,
          chunk, blocked, chunkAxis,
          [&](OpBuilder &nested, IRMapping &mapping, ValueRange coordinates,
              Value valid) -> LogicalResult {
            for (unsigned axis : ownedAxes)
              ownedCoordinates[axis] = coordinates[axis];
            auto coordinateType = cast<FragmentType>(coordinates[chunkAxis].getType());
            auto tailType = FragmentType::get(kernel.getContext(), nested.getI1Type(),
                coordinateType.getShape(), coordinateType.getAxisMaps(),
                coordinateType.getValidity(), coordinateType.getOwner());
            Value end = nested.create<BroadcastOp>(gather.getLoc(), coordinateType,
                                                   ranges[chunkAxis].getLogicalStop());
            options.segmentTail = nested.create<CompareOp>(gather.getLoc(), tailType,
                coordinates[chunkAxis], end, ComparePredicate::Lt);
            FailureOr<Value> value = materializeReplayedValue(
                nested, gather.getLoc(), source, axis, chunkExtent, mapping,
                insertionAnchor, options);
            if (failed(value))
              return gather.emitOpError("retained gather source has no bounded producer replay");
            nested.create<StoreOp>(gather.getLoc(), workspace, coordinates, *value,
                                   valid, sourceAxes);
            return success();
          })))
    return failure();
  for (GatherOp reader : readers) {
    builder.setInsertionPoint(reader);
    auto result = dyn_cast<FragmentType>(reader.getResult().getType());
    SmallVector<Value> coordinates(reader.getCoordinates());
    Value valid = reader.getValid();
    for (unsigned axis : ownedAxes) {
      auto indexType = FragmentType::get(result.getContext(), builder.getIndexType(),
          result.getShape(), result.getAxisMaps(), result.getValidity(), result.getOwner());
      auto predicate = FragmentType::get(result.getContext(), builder.getI1Type(),
          result.getShape(), result.getAxisMaps(), result.getValidity(), result.getOwner());
      unsigned position = llvm::find(reader.getSourceAxes(), axis) - reader.getSourceAxes().begin();
      auto ordinal = ordinalRange(reader.getCoordinates()[position]);
      auto identity = builder.getArrayAttr({ReshapeGroupAttr::get(
          kernel.getContext(), builder.getDenseI64ArrayAttr({0}),
          builder.getDenseI64ArrayAttr({0}))});
      Value rebound = builder.create<ReshapeOp>(reader.getLoc(),
          ordinal.getResult().getType(), ownedCoordinates[axis], identity);
      auto projected = projectPhysicalValueToSchema(builder, reader.getLoc(), rebound, indexType);
      if (failed(projected))
        return reader.emitOpError("workspace reader lost its owned range projection");
      coordinates[position] = *projected;
      Value end = builder.create<SplatOp>(reader.getLoc(), indexType, ranges[axis].getLogicalStop());
      Value tail = builder.create<CompareOp>(reader.getLoc(), predicate, *projected, end,
                                             ComparePredicate::Lt);
      valid = valid ? Value(builder.create<BinaryOp>(reader.getLoc(), predicate, valid,
                         tail, BinaryOperator::LogicalAnd)) : tail;
    }
    Value fill = reader.getFill();
    if (valid && !fill) {
      if (result) {
        auto zero = materializeZeroFragment(builder, reader.getLoc(), result);
        if (failed(zero))
          return failure();
        fill = *zero;
      } else {
        fill = builder.create<arith::ConstantOp>(reader.getLoc(),
            builder.getZeroAttr(reader.getResult().getType()));
      }
    }
    auto load = builder.create<LoadOp>(reader.getLoc(), reader.getResult().getType(),
        workspace, coordinates, valid, fill,
        reader.getSourceAxes());
    if (Attribute origin = reader->getAttr(originAttr))
      load->setAttr(originAttr, origin);
    reader.getResult().replaceAllUsesWith(load.getResult());
    reader.erase();
  }
  eraseDeadPhysicalValues(kernel);
  return true;
}

} // namespace

static LogicalResult materializeRetainedValuesImpl(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  while (true) {
    SmallVector<GatherOp> gathers;
    kernel.walk([&](GatherOp gather) { gathers.push_back(gather); });
    bool changed = false;
    for (GatherOp gather : gathers) {
      FailureOr<bool> result = materializeRetainedGather(gather, kernel);
      if (failed(result))
        return failure();
      if (*result) {
        changed = true;
        break;
      }
    }
    if (changed)
      continue;
    SmallVector<scf::ForOp> loops;
    kernel.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops)
      if (materializeLoopState(loop, kernel)) {
        changed = true;
        break;
      }
    if (changed)
      continue;
    auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
    if (!llvm::all_of(space, [](Attribute attribute) {
          auto extent = cast<PhysicalExprAttr>(attribute);
          return extent.getKind() == PhysicalExprKind::Constant &&
                 extent.getValue() == 1;
        }))
      return success();
    SmallVector<StoreOp> stores;
    kernel.walk([&](StoreOp store) { stores.push_back(store); });
    for (StoreOp store : stores) {
      FailureOr<bool> result = materializeRetainedStore(store, kernel);
      if (failed(result))
        return failure();
      if (*result) {
        changed = true;
        break;
      }
    }
    if (!changed)
      return success();
  }
}

LogicalResult materializeRetainedValues(ModuleOp module) {
  if (failed(materializeRetainedValuesImpl(module))) return failure();
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : closeValueRelations(*kernel);
}

} // namespace intent::gpu
