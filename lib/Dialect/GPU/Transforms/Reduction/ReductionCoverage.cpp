#include "ReductionRealization.h"
#include "ReductionParameters.h"
#include "ReductionValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::gpu::reduction {
namespace {

FailureOr<Value> clonePaddedProducer(
    OpBuilder &builder, Location location, Value value,
    std::optional<unsigned> projectedAxis,
    PhysicalSourceAxis reductionSource,
    ArrayRef<MakeRangeOp> selectedRanges,
    PhysicalExprAttr logicalExtent, PhysicalExprAttr physicalExtent,
    Value physicalExtentValue, IRMapping &mapping,
    SmallVectorImpl<Value> &tailPredicates) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return value;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalRangeAxisFact selected =
      PhysicalProgramAnalysis(kernel).rangeAxes(value, selectedRanges);
  if (!selected.isExact())
    return failure();
  if (selected.fragmentAxes.size() > 1)
    return failure();
  if (!selected.fragmentAxes.empty() && projectedAxis &&
      selected.fragmentAxes.front() != *projectedAxis)
    return failure();
  if (!projectedAxis && !selected.fragmentAxes.empty())
    projectedAxis = selected.fragmentAxes.front();
  if (!projectedAxis)
    return value;
  unsigned reductionAxis = *projectedAxis;
  if (reductionAxis >= fragment.getShape().size())
    return failure();
  if (fragment.getShape()[reductionAxis] != logicalExtent)
    return value;
  Operation *producer = value.getDefiningOp();
  if (!producer || producer->getNumResults() != 1 ||
      (producer->getNumRegions() != 0 && !isa<ReduceOp>(producer)))
    return producer
               ? (producer->emitOpError(
                      "padded reduction producer is not a replayable single-result operation"),
                  FailureOr<Value>(failure()))
               : FailureOr<Value>(failure());

  if (auto range = dyn_cast<MakeRangeOp>(producer)) {
    auto paddedType = replaceExtent(fragment, reductionAxis, physicalExtent);
    auto padded = builder.create<MakeRangeOp>(
        location, paddedType, range.getStart(), physicalExtentValue,
        range.getStep(), range.getLogicalStart(), range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    inheritRangeAuthority(padded, range);
    Value logicalLength = builder.create<arith::ConstantIndexOp>(
        location, logicalExtent.getValue());
    Value distance = builder.create<BinaryOp>(
        location, builder.getIndexType(), logicalLength, range.getStep(),
        BinaryOperator::Multiply);
    Value stop = builder.create<BinaryOp>(location, builder.getIndexType(),
                                          range.getStart(), distance,
                                          BinaryOperator::Add);
    Value stopFragment =
        builder.create<BroadcastOp>(location, paddedType, stop);
    auto predicateType = FragmentType::get(
        builder.getContext(), builder.getI1Type(), paddedType.getShape(),
        paddedType.getAxisMaps(), paddedType.getValidity(),
        paddedType.getOwner());
    Value predicate = builder.create<CompareOp>(
        location, predicateType, padded.getResult(), stopFragment,
        ComparePredicate::Lt);
    tailPredicates.push_back(predicate);
    mapping.map(value, padded.getResult());
    return padded.getResult();
  }

  if (auto broadcast = dyn_cast<BroadcastOp>(producer)) {
    auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
    auto paddedType = replaceExtent(fragment, reductionAxis, physicalExtent);
    if (!input) {
      auto padded = builder.create<BroadcastOp>(location, paddedType,
                                                broadcast.getValue());
      if (Attribute origin = broadcast->getAttr(originAttr))
        padded->setAttr(originAttr, origin);
      mapping.map(value, padded.getResult());
      return padded.getResult();
    }
    BroadcastProjection projection =
        queryAxisProjection(input, fragment);
    if (!projection.isExact() ||
        reductionAxis >= projection.targetToSource.size()) {
      InFlightDiagnostic diagnostic = broadcast.emitOpError(
          "padded broadcast has no exact source-axis projection");
      diagnostic << "; input=" << broadcast.getValue().getType()
                 << "; result=" << fragment
                 << "; reduction_axis=" << reductionAxis;
      return failure();
    }
    Value replayed = broadcast.getValue();
    if (std::optional<unsigned> inputAxis =
            projection.targetToSource[reductionAxis]) {
      auto inputExtent =
          cast<PhysicalExprAttr>(input.getShape()[*inputAxis]);
      bool expandsSingleton =
          inputExtent.getKind() ==
              PhysicalExprKind::Constant &&
          inputExtent.getValue() == 1 &&
          input.getShape()[*inputAxis] != fragment.getShape()[reductionAxis];
      if (!expandsSingleton) {
        auto inputMapping =
            cast<AxisMapAttr>(input.getAxisMaps()[*inputAxis]);
        FailureOr<Value> replacement = clonePaddedProducer(
            builder, location, broadcast.getValue(), *inputAxis,
            PhysicalSourceAxis{inputMapping.getSourceId(),
                               inputMapping.getSourceAxis(),
                               inputMapping.getDerived()},
            selectedRanges, logicalExtent, physicalExtent,
            physicalExtentValue, mapping, tailPredicates);
        if (failed(replacement))
          return failure();
        replayed = *replacement;
      }
    }
    auto padded =
        builder.create<BroadcastOp>(location, paddedType, replayed);
    if (Attribute origin = broadcast->getAttr(originAttr))
      padded->setAttr(originAttr, origin);
    mapping.map(value, padded.getResult());
    return padded.getResult();
  }

  if (auto load = dyn_cast<LoadOp>(producer)) {
    Operation *anchor = builder.getInsertionPoint() ==
                                builder.getInsertionBlock()->end()
                            ? nullptr
                            : &*builder.getInsertionPoint();
    if (anchor != load.getOperation() && !canReplayReadAt(load, anchor)) {
      // Keep a read that crosses a possible writer at its original definition.
      // Its coordinate dependencies already dominate that point; a separate
      // mapping prevents later padded coordinates from moving before them.
      OpBuilder snapshotBuilder(load);
      Value snapshotExtent = snapshotBuilder.create<arith::ConstantIndexOp>(
          load.getLoc(), physicalExtent.getValue());
      IRMapping snapshotMapping;
      SmallVector<Value> snapshotTails;
      FailureOr<Value> snapshot = clonePaddedProducer(
          snapshotBuilder, load.getLoc(), value, projectedAxis, reductionSource,
          selectedRanges, logicalExtent, physicalExtent, snapshotExtent,
          snapshotMapping, snapshotTails);
      if (failed(snapshot))
        return failure();
      mapping.map(value, *snapshot);
      tailPredicates.append(snapshotTails.begin(), snapshotTails.end());
      return *snapshot;
    }
    SmallVector<Value> coordinates;
    for (Value coordinate : load.getCoordinates()) {
      FailureOr<Value> replayed = clonePaddedProducer(
          builder, location, coordinate, std::nullopt, reductionSource,
          selectedRanges,
          logicalExtent,
          physicalExtent,
          physicalExtentValue, mapping, tailPredicates);
      if (failed(replayed))
        return failure();
      coordinates.push_back(*replayed);
    }
    Value valid;
    if (load.getValid()) {
      FailureOr<Value> replayed = clonePaddedProducer(
          builder, location, load.getValid(), reductionAxis, reductionSource,
          selectedRanges,
          logicalExtent,
          physicalExtent,
          physicalExtentValue, mapping, tailPredicates);
      if (failed(replayed))
        return failure();
      valid = *replayed;
    }
    Value fill;
    if (load.getFill()) {
      FailureOr<Value> replayed = clonePaddedProducer(
          builder, location, load.getFill(), reductionAxis, reductionSource,
          selectedRanges,
          logicalExtent,
          physicalExtent,
          physicalExtentValue, mapping, tailPredicates);
      if (failed(replayed))
        return failure();
      fill = *replayed;
    }
    auto paddedType = replaceExtent(fragment, reductionAxis, physicalExtent);
    if (tailPredicates.empty())
      return load.emitOpError(
                 "padded reduction load has no logical tail predicate"),
             failure();
    Value tail;
    for (Value base : tailPredicates) {
      FailureOr<Value> current = predicateForReductionSource(
          builder, location, base, paddedType, reductionAxis);
      if (failed(current))
        return failure();
      tail = tail ? Value(builder.create<BinaryOp>(
                        location, current->getType(), tail, *current,
                        BinaryOperator::LogicalAnd))
                  : *current;
    }
    auto predicateType = cast<FragmentType>(tail.getType());
    if (valid) {
      Type element = valid.getType();
      if (auto validFragment = dyn_cast<FragmentType>(element))
        element = validFragment.getElementType();
      if (!element.isInteger(1))
        return load.emitOpError(
                   "padded reduction replay produced non-predicate validity"),
               failure();
      if (valid.getType() != predicateType)
        valid = builder.create<BroadcastOp>(location, predicateType, valid);
      valid = builder.create<BinaryOp>(location, predicateType, valid, tail,
                                       BinaryOperator::LogicalAnd);
    } else {
      valid = tail;
    }
    if (!fill) {
      FailureOr<Value> zero =
          materializeZeroFragment(builder, location, paddedType);
      if (failed(zero))
        return failure();
      fill = *zero;
    } else if (fill.getType() != paddedType)
      fill = builder.create<BroadcastOp>(location, paddedType, fill);
    auto padded = builder.create<LoadOp>(
        location, paddedType, load.getResource(), coordinates, valid, fill,
        load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      padded->setAttr(originAttr, origin);
    mapping.map(value, padded.getResult());
    return padded.getResult();
  }

  for (Value operand : producer->getOperands()) {
    std::optional<unsigned> operandAxis;
    if (auto operandType = dyn_cast<FragmentType>(operand.getType())) {
      BroadcastProjection projection = queryAxisProjection(operandType, fragment);
      if (projection.isExact() &&
          reductionAxis < projection.targetToSource.size())
        operandAxis = projection.targetToSource[reductionAxis];
      if (operandAxis) {
        auto operandExtent =
            cast<PhysicalExprAttr>(operandType.getShape()[*operandAxis]);
        if (operandExtent.getKind() ==
                PhysicalExprKind::Constant &&
            operandExtent.getValue() == 1 &&
            operandType.getShape()[*operandAxis] !=
                fragment.getShape()[reductionAxis])
          operandAxis.reset();
      }
    }
    FailureOr<Value> replayed = clonePaddedProducer(
        builder, location, operand, operandAxis, reductionSource, selectedRanges,
        logicalExtent,
        physicalExtent,
        physicalExtentValue, mapping, tailPredicates);
    if (failed(replayed))
      return failure();
    if (!mapping.lookupOrNull(operand) && *replayed != operand)
      mapping.map(operand, *replayed);
  }
  auto paddedType = replaceExtent(fragment, reductionAxis, physicalExtent);
  auto cloned = cloneWithPhysicalSchema(builder, producer, mapping,
      [&](Value original) -> Type {
        if (original == value) return paddedType;
        auto type = dyn_cast<FragmentType>(original.getType());
        if (!type) return original.getType();
        auto axis = queryFragmentAxis(type, reductionSource);
        if (!axis.isExact() || type.getShape()[axis.fragmentAxis] != logicalExtent)
          return type;
        return replaceExtent(type, axis.fragmentAxis, physicalExtent);
      });
  if (failed(cloned)) return failure();
  return mapping.lookup(value);
}

} // namespace

FailureOr<bool> realizeStaticPaddingReduce(ReduceOp reduce,
                                           func::FuncOp kernel) {
  if (reduce.getAxes().size() != 1 || reduce.getSources().size() == 0)
    return false;
  int64_t reductionAxis = reduce.getAxes().front();
  if (reductionAxis < 0)
    return false;

  PhysicalExprAttr logicalExtent;
  for (Value source : reduce.getSources()) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment ||
        reductionAxis >= static_cast<int64_t>(fragment.getShape().size()))
      return false;
    auto extent =
        dyn_cast<PhysicalExprAttr>(fragment.getShape()[reductionAxis]);
    if (!extent || extent.getKind() !=
                       PhysicalExprKind::Constant)
      return false;
    if (logicalExtent && logicalExtent != extent)
      return false;
    logicalExtent = extent;
  }
  if (!logicalExtent || logicalExtent.getValue() <= 0)
    return false;
  PhysicalExprAttr physicalExtent = nextPowerOfTwo(logicalExtent);
  if (physicalExtent == logicalExtent)
    return false;

  SmallVector<SmallVector<MakeRangeOp>> componentRanges;
  for (Value source : reduce.getSources()) {
    auto fragment = cast<FragmentType>(source.getType());
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact reductionRanges =
        analysis.axisRanges(source, static_cast<unsigned>(reductionAxis));
    FailureOr<MakeRangeOp> authority =
        queryExactLogicalRange(reductionRanges);
    if (failed(authority) &&
        reductionRanges.state != PhysicalFactState::Unknown &&
        reductionRanges.blockers.empty()) {
      PhysicalLockstepTraversalFact traversal =
          analysis.lockstepRanges(reductionRanges.roots);
      if (traversal.isExact())
        authority = traversal.authority;
    }
    if (failed(authority) && reductionRanges.roots.empty()) {
      auto mapping = cast<AxisMapAttr>(
          fragment.getAxisMaps()[static_cast<unsigned>(reductionAxis)]);
      PhysicalSourceAxis reductionSource{mapping.getSourceId(),
                                         mapping.getSourceAxis(),
                                         mapping.getDerived()};
      PhysicalRangeFact programRanges =
          analysis.programRanges(reductionSource);
      PhysicalRangeFact dimensionRanges;
      dimensionRanges.state = PhysicalFactState::Exact;
      for (MakeRangeOp range : programRanges.roots) {
        FailureOr<int64_t> dimension = queryRangeDimension(range);
        if (succeeded(dimension) &&
            *dimension == mapping.getDimensionId())
          dimensionRanges.roots.push_back(range);
      }
      if (dimensionRanges.roots.empty())
        dimensionRanges.state = PhysicalFactState::Unknown;
      else if (dimensionRanges.roots.size() > 1)
        dimensionRanges.state = PhysicalFactState::Ambiguous;
      dimensionRanges.unitStep =
          !dimensionRanges.roots.empty() &&
          llvm::all_of(dimensionRanges.roots, isUnitStepRange);
      authority = queryExactLogicalRange(dimensionRanges);
      if (succeeded(authority))
        reductionRanges = std::move(dimensionRanges);
    }
    SmallVector<MakeRangeOp> ranges(reductionRanges.roots.begin(),
                                    reductionRanges.roots.end());
    llvm::sort(ranges, [](MakeRangeOp lhs, MakeRangeOp rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());
    if (failed(authority)) {
      InFlightDiagnostic diagnostic = reduce.emitOpError(
          "static non-power-of-two reduction axis has no exact range authority");
      diagnostic << "; range_state="
                 << static_cast<unsigned>(reductionRanges.state)
                 << ", roots=" << reductionRanges.roots.size()
                 << ", blockers=" << reductionRanges.blockers.size()
                 << ", source_type=" << source.getType();
      if (Operation *definition = source.getDefiningOp()) {
        diagnostic << ", source_def=" << definition->getName();
        for (Value operand : definition->getOperands())
          if (Operation *operandDefinition = operand.getDefiningOp())
            diagnostic << ", operand_def=" << operandDefinition->getName();
      }
      for (MakeRangeOp root : reductionRanges.roots) {
        FailureOr<int64_t> dimension = queryRangeDimension(root);
        diagnostic << ", root=(source_id=" << root.getSourceId()
                   << ",source_axis=" << root.getSourceAxis()
                   << ",dimension="
                   << (succeeded(dimension) ? *dimension : -1) << ")";
      }
      for (Operation *blocker : reductionRanges.blockers)
        diagnostic << ", blocker=" << blocker->getName();
      return failure();
    }
    componentRanges.push_back(std::move(ranges));
  }
  if (llvm::any_of(componentRanges,
                   [](ArrayRef<MakeRangeOp> ranges) { return ranges.empty(); }))
    return reduce.emitOpError(
               "static non-power-of-two reduction source has no exact logical range"),
           failure();

  OpBuilder builder(reduce);
  Value physicalExtentValue = builder.create<arith::ConstantIndexOp>(
      reduce.getLoc(), physicalExtent.getValue());
  for (unsigned component = 0; component < reduce.getSources().size(); ++component) {
    auto originalType =
        cast<FragmentType>(reduce.getSources()[component].getType());
    auto sourceMap = cast<AxisMapAttr>(
        originalType.getAxisMaps()[static_cast<unsigned>(reductionAxis)]);
    PhysicalSourceAxis reductionSource{sourceMap.getSourceId(),
                                       sourceMap.getSourceAxis(),
                                       sourceMap.getDerived()};
    PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayAt(
        reduce.getSources()[component], reductionSource,
        PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true, reduce, IRMapping{});
    if (!replay.isReplayable()) {
      InFlightDiagnostic diagnostic = reduce.emitOpError(
          "static reduction producer has no exact shared replay fact");
      for (Operation *blocker : replay.blockers)
        diagnostic << "; blocker=" << blocker->getName();
      return failure();
    }
    IRMapping mapping;
    SmallVector<Value> tailPredicates;
    FailureOr<Value> source = clonePaddedProducer(
        builder, reduce.getLoc(), reduce.getSources()[component],
        static_cast<unsigned>(reductionAxis), reductionSource,
        componentRanges[component], logicalExtent, physicalExtent,
        physicalExtentValue, mapping,
        tailPredicates);
    if (failed(source) || tailPredicates.empty()) {
      PhysicalRangeAxisFact selected =
          PhysicalProgramAnalysis(kernel).rangeAxes(
              reduce.getSources()[component], componentRanges[component]);
      InFlightDiagnostic diagnostic = reduce.emitOpError(
          "static reduction producer cannot be replayed over its padded extent");
      diagnostic << "; selected_state=" << static_cast<unsigned>(selected.state)
                 << ", selected_axes=" << selected.fragmentAxes.size()
                 << ", selected_ranges=" << componentRanges[component].size()
                 << ", replay_failed=" << failed(source)
                 << ", tail_count=" << tailPredicates.size();
      return failure();
    }
    reduce->setOperand(component, *source);
  }
  eraseDeadPhysicalValues(kernel);
  return true;
}

LogicalResult neutralizeReductionTails(ReduceOp reduce, func::FuncOp kernel) {
  if (reduce.getAxes().size() != 1)
    return success();
  unsigned axis = reduce.getAxes().front();
  DominanceInfo dominance(kernel);
  for (unsigned component = 0; component < reduce.getSources().size(); ++component) {
    Value source = reduce.getSources()[component];
    auto type = dyn_cast<FragmentType>(source.getType());
    if (!type || axis >= type.getShape().size())
      continue;
    if (auto loop = source.getDefiningOp<scf::ForOp>()) {
      auto traversals = loop->getAttrOfType<ArrayAttr>(reductionSourcesAttr);
      auto mapping = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
      auto traversal = PhysicalSourceAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDerived());
      unsigned result = cast<OpResult>(source).getResultNumber();
      // Reduction blocking pads each chunk with its identity before updating
      // this carry. Its completed lanes no longer share an inner chunk's tail.
      if (traversals &&
          llvm::equal(reduce.getSources(),
                      loop.getResults()) &&
          llvm::is_contained(traversals, Attribute(traversal)) &&
          queryLaunchExpression(loop.getStep()) == type.getShape()[axis] &&
          sameScalarValue(loop.getInitArgs()[result],
                          reduce.getIdentities()[component]))
        continue;
    }
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact ranges = analysis.axisRanges(source, axis);
    auto constant = [](Value value) -> std::optional<int64_t> {
      auto expression = queryLaunchExpression(value);
      return expression ? constantPhysicalExpression(expression) : std::nullopt;
    };
    bool needsTail = llvm::any_of(ranges.roots, [&](MakeRangeOp range) {
      if (range.getResult().getType().getShape()[0] != type.getShape()[axis])
        return false;
      auto start = constant(range.getLogicalStart());
      auto stop = constant(range.getLogicalStop());
      auto extent = constant(range.getExtent());
      return !isUnitStepRange(range) || !start || !stop || !extent ||
             !samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()) ||
             static_cast<__int128>(*stop) - *start != *extent;
    });
    if (!needsTail)
      continue;
    if ((!ranges.isExact() && !analysis.lockstepRanges(ranges.roots).isExact() &&
         !sameFullOrdinalTraversal(ranges.roots)) ||
        !ranges.blockers.empty()) {
      auto diagnostic = reduce.emitOpError(
          "padded reduction has no exact logical member traversal");
      diagnostic << "; source=" << source << "; axis=" << axis;
      for (Operation *blocker : ranges.blockers)
        diagnostic << "; blocker=" << *blocker;
      return failure();
    }
    OpBuilder builder(reduce);
    Value tail;
    for (MakeRangeOp range : ranges.roots) {
      if (range.getResult().getType().getShape()[0] != type.getShape()[axis])
        continue;
      if (!isUnitStepRange(range))
        return reduce.emitOpError(
            "padded reduction has no dominating coordinate predicate");
      if (!dominance.dominates(range.getOperation(), reduce.getOperation())) {
        if (!llvm::all_of(range->getOperands(), [&](Value operand) {
              return dominance.dominates(operand, reduce.getOperation());
            }))
          return reduce.emitOpError(
              "padded reduction coordinate depends on an inner traversal");
        range = cast<MakeRangeOp>(builder.clone(*range.getOperation()));
      }
      auto coordinates = range.getResult().getType();
      auto predicate = FragmentType::get(kernel.getContext(), builder.getI1Type(),
          coordinates.getShape(), coordinates.getAxisMaps(),
          coordinates.getValidity(), coordinates.getOwner());
      Value end = builder.create<SplatOp>(reduce.getLoc(), coordinates,
                                          range.getLogicalStop());
      Value valid = builder.create<CompareOp>(reduce.getLoc(), predicate,
          range.getResult(), end, ComparePredicate::Lt);
      auto projected = predicateForReductionSource(builder, reduce.getLoc(), valid,
                                                   type, axis);
      if (failed(projected))
        return reduce.emitOpError("reduction tail has no physical axis projection");
      tail = tail ? Value(builder.create<BinaryOp>(reduce.getLoc(), projected->getType(),
                     tail, *projected, BinaryOperator::LogicalAnd)) : *projected;
    }
    if (!tail)
      continue;
    Value identity = reduce.getIdentities()[component];
    auto projected = projectPhysicalValueToSchema(builder, reduce.getLoc(), identity, type);
    if (failed(projected))
      return reduce.emitOpError("reduction identity cannot neutralize physical padding");
    reduce->setOperand(component, builder.create<SelectOp>(
        reduce.getLoc(), type, tail, source, *projected).getResult());
  }
  return success();
}

FailureOr<bool> realizeFullCoverageReduce(ReduceOp reduce,
                                          func::FuncOp kernel) {
  if (reduce.getAxes().size() != 1 || reduce.getSources().size() == 0)
    return false;
  int64_t reductionAxis = reduce.getAxes().front();
  if (reductionAxis < 0)
    return false;

  PhysicalExprAttr extent;
  SmallVector<PhysicalSourceAxis> sourceAxes;
  for (Value source : reduce.getSources()) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment ||
        reductionAxis >= static_cast<int64_t>(fragment.getShape().size()))
      return false;
    FailureOr<AxisMapAttr> mapping = queryAxisMap(fragment, reductionAxis);
    if (failed(mapping))
      return false;
    PhysicalExprAttr current =
        cast<PhysicalExprAttr>(fragment.getShape()[reductionAxis]);
    if (extent && extent != current)
      return false;
    extent = current;
    sourceAxes.push_back(sourceAxisIdentity(*mapping));
  }
  if (!extent || extent.getKind() !=
                     PhysicalExprKind::Parameter)
    return false;
  llvm::DenseMap<PhysicalSourceAxis, MakeRangeOp> ranges;
  PhysicalProgramAnalysis analysis(kernel);
  for (auto [source, sourceAxis] : llvm::zip(
           reduce.getSources(), sourceAxes)) {
    PhysicalRangeFact fact = analysis.axisRanges(source, reductionAxis);
    for (MakeRangeOp candidate : fact.roots) {
      auto found = ranges.find(sourceAxis);
      if (found != ranges.end() &&
          !sameLogicalRange(found->second, candidate))
        return false;
      ranges[sourceAxis] = candidate;
    }
  }
  if (ranges.empty())
    return false;
  MakeRangeOp range = ranges.begin()->second;
  if (range->hasAttr(sourceSubregionAttr))
    return false;
  FailureOr<ParameterAttr> parameter = fullCoverageParameter(kernel, extent);
  if (failed(parameter))
    return false;
  if (llvm::any_of(ranges, [](const auto &entry) {
        return !isUnitStepRange(entry.second);
      }))
    return reduce.emitOpError(
               "full-coverage reduction tail requires unit-step coordinates"),
           failure();
  Value logicalExtent = range.getExtent();
  if (isCompileTimeValue(logicalExtent)) {
    FailureOr<int64_t> sourceDimension = queryRangeDimension(range);
    auto coverageDimension = parameter->getBinding().getDimension();
    if (range->hasAttr(sourceSubregionAttr) || failed(sourceDimension) ||
        !coverageDimension ||
        *sourceDimension != coverageDimension.getInt())
      return false;
    FailureOr<Value> launchExtent =
        dimensionArgument(kernel, *sourceDimension);
    if (failed(launchExtent))
      return false;
    logicalExtent = *launchExtent;
  }

  auto coverageDimension = parameter->getBinding().getDimension();
  if (!coverageDimension ||
      failed(bindFullCoverageDimension(
          kernel, coverageDimension.getInt(), parameterValue(kernel, *parameter))))
    return failure();

  OpBuilder builder(reduce);
  for (unsigned component = 0; component < reduce.getSources().size(); ++component) {
    Value source = reduce.getSources()[component];
    auto found = ranges.find(sourceAxes[component]);
    if (found == ranges.end())
      return false;
    MakeRangeOp componentRange = found->second;
    auto coordinateType =
        cast<FragmentType>(componentRange.getResult().getType());
    auto sourceType = cast<FragmentType>(source.getType());
    Value stop = builder.create<BinaryOp>(
        reduce.getLoc(), builder.getIndexType(), componentRange.getStart(),
        logicalExtent, BinaryOperator::Add);
    Value stopFragment =
        builder.create<BroadcastOp>(reduce.getLoc(), coordinateType, stop);
    auto coordinatePredicate = FragmentType::get(
        reduce.getContext(), builder.getI1Type(), coordinateType.getShape(),
        coordinateType.getAxisMaps(), coordinateType.getValidity(),
        coordinateType.getOwner());
    Value coordinateValid = builder.create<CompareOp>(
        reduce.getLoc(), coordinatePredicate, componentRange.getResult(),
        stopFragment, ComparePredicate::Lt);
    auto predicateType = FragmentType::get(
        reduce.getContext(), builder.getI1Type(), sourceType.getShape(),
        sourceType.getAxisMaps(), sourceType.getValidity(), sourceType.getOwner());
    Value valid = coordinateValid;
    if (valid.getType() != predicateType)
      valid = builder.create<BroadcastOp>(reduce.getLoc(), predicateType, valid);
    Value identity = reduce.getIdentities()[component];
    if (identity.getType() != sourceType)
      identity = builder.create<BroadcastOp>(reduce.getLoc(), sourceType, identity);
    Value selected = builder.create<SelectOp>(reduce.getLoc(), sourceType, valid,
                                              source, identity);
    reduce->setOperand(component, selected);
  }
  eraseDeadPhysicalValues(kernel);
  return true;
}

} // namespace intent::gpu::reduction
