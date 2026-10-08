#include "ReductionAnalysis.h"
#include "../Value/ReplayPolicy.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include <functional>

using namespace mlir;

namespace intent::gpu::reduction {

bool isCompileTimeExtent(PhysicalExprAttr expression) {
  auto kind = expression.getKind();
  if (kind == PhysicalExprKind::Constant || kind == PhysicalExprKind::Parameter)
    return true;
  if (kind == PhysicalExprKind::Dimension ||
      kind == PhysicalExprKind::ScalarABI)
    return false;
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isCompileTimeExtent(cast<PhysicalExprAttr>(operand));
  });
}

bool isCompileTimeValue(Value value) {
  return value.getDefiningOp<arith::ConstantOp>() ||
         value.getDefiningOp<ParameterOp>() ||
         value.getDefiningOp<PhysicalExprOp>();
}

bool exceedsRegisterFile(Value source, func::FuncOp kernel) {
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || !capabilities || capabilities.getRegistersPerUnit() <= 0)
    return false;
  int64_t budget = capabilities.getRegistersPerUnit();
  auto registers = minimumFragmentRegisterFootprint(kernel, source, budget,
      FragmentFootprintScope::PhysicalShape);
  // Even the smallest source must leave space for the reduction state.
  // Larger concrete tuples are filtered after all parameters are bound.
  return registers && *registers >= budget;
}

bool shouldTileReductionSources(ReduceOp reduce, func::FuncOp kernel) {
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0 ||
      reduce.getAxes().empty()) return false;
  int64_t budget = capabilities.getRegistersPerUnit();
  __int128 footprint = 0;
  llvm::DenseSet<Value> sources;
  for (Value source : reduce.getSources()) {
    // A full/splat scalar is not another fragment-sized materialization. Nor
    // do repeated tuple operands require independent physical payloads.
    if (uniformScalarSource(source)) {
      // The optional schedule must use the same scalar frontier supported by
      // the realizer, even when uniform analysis can see through more views.
      if (failed(scalarSource(source))) return false;
      continue;
    }
    if (!sources.insert(source).second) continue;
    auto words = minimumFragmentRegisterFootprint(
        kernel, source, budget, FragmentFootprintScope::PhysicalShape);
    if (!words) return false;
    footprint += *words;
  }
  if (sources.size() < 2 || footprint < budget) return false;

  ReplayPolicy reuse(kernel, reduce.getSources(), {reduce.getOperation()});
  PhysicalProgramAnalysis analysis(kernel);
  for (Value source : sources) {
    if (!reuse.removesProducer(source) || reuse.duplicatesExpensiveWork(source))
      return false;
    auto type = dyn_cast<FragmentType>(source.getType());
    if (!type) return false;
    for (int64_t axis : reduce.getAxes()) {
      if (axis < 0 || axis >= static_cast<int64_t>(type.getShape().size()))
        return false;
      auto relation = queryAxisMap(type, axis);
      auto plan = analyzeSource(source, axis, /*diagnose=*/false);
      if (failed(relation) || failed(plan) || plan->roots.empty() ||
          !analysis.replayAt(source, sourceAxisIdentity(*relation),
              PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true,
              reduce, IRMapping{}, relation->getDimensionId()).isReplayable())
        return false;
      for (LoadOp load : plan->roots) {
        auto access = analyzeRoot(load, plan->ranges, plan->reductionAxis,
                                  /*diagnose=*/false);
        if (failed(access) || !*access || !reuse.removesProducer(load.getResult()))
          return false;
      }
    }
  }
  return true;
}

bool canChunkReductionOuter(ReduceOp reduce, func::FuncOp kernel,
                            unsigned outerAxis) {
  if (!hasNonUnitFreeAxis(reduce) || failed(proveLaneWiseHelper(reduce.getCombine())))
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  ReplayPolicy reuse(kernel, reduce.getSources(), {reduce.getOperation()});
  SmallVector<MakeRangeOp> ranges;
  bool hasRead = false;
  for (Value source : reduce.getSources()) {
    // Uniform components participate in the very same member set, but need no
    // fictitious memory/range root. The realizer broadcasts their actual scalar
    // and applies this outer slice's identity predicate component by component.
    if (uniformScalarSource(source)) {
      if (failed(scalarSource(source))) return false;
      continue;
    }
    auto plan = analyzeSource(source, outerAxis, /*diagnose=*/false);
    if (failed(plan) || plan->roots.empty() || !reuse.removesProducer(source) ||
        reuse.duplicatesExpensiveWork(source)) return false;
    auto type = cast<FragmentType>(source.getType());
    auto axis = cast<AxisMapAttr>(type.getAxisMaps()[outerAxis]);
    auto replay = analysis.replayAt(source, plan->sourceIdentity,
        PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true, reduce,
        IRMapping{}, axis.getDimensionId());
    if (!replay.isReplayable() || llvm::any_of(replay.accesses,
        [](Operation *access) { return !isa<LoadOp>(access); })) return false;
    for (MakeRangeOp range : plan->ranges) {
      if (range->hasAttr(sourceSubregionAttr) || !isUnitStepRange(range)) return false;
      ranges.push_back(range);
    }
    for (LoadOp load : plan->roots) {
      auto access = analyzeRoot(load, plan->ranges, plan->reductionAxis, /*diagnose=*/false);
      if (failed(access) || !*access || (*access)->range->hasAttr(sourceSubregionAttr) ||
          !reuse.removesProducer(load.getResult())) return false;
      ranges.push_back((*access)->range);
      hasRead = true;
    }
  }
  return hasRead && !ranges.empty() && analysis.lockstepRanges(ranges).isExact();
}

bool requiresPhysicalRealization(ReduceOp reduce) {
  auto kernel = reduce->getParentOfType<func::FuncOp>();
  if (!kernel)
    return true;
  PhysicalProgramAnalysis analysis(kernel);
  for (Value source : reduce.getSources()) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment)
      continue;
    for (int64_t axis : reduce.getAxes()) {
      if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
        continue;
      PhysicalAxisRealizationFact fact =
          analysis.axisRealization(source, static_cast<unsigned>(axis));
      if (fact.constructionScalarSeed ||
          (fact.isExact() && !fact.physicalized)) {
        auto extent = constantPhysicalExpression(
            cast<PhysicalExprAttr>(fragment.getShape()[axis]));
        bool pairedComplete = extent && succeeded(scalarSource(source)) &&
            llvm::any_of(reduce.getSources(),
                         [&](Value peer) {
          auto peerType = dyn_cast<FragmentType>(peer.getType());
          if (!peerType || peerType.getShape() != fragment.getShape() ||
              peerType.getAxisMaps() != fragment.getAxisMaps() ||
              peerType.getValidity() != fragment.getValidity() ||
              peerType.getOwner() != fragment.getOwner())
            return false;
          auto coverage = analysis.axisRealization(peer, axis);
          return coverage.isExact() && coverage.physicalized &&
                 !coverage.constructionScalarSeed &&
                 !coverage.roots.empty() &&
                 llvm::all_of(coverage.roots, [&](MakeRangeOp range) {
            return constantLogicalRangeCardinality(range) == extent &&
                   samePhysicalScalarExpression(range.getStart(),
                                                range.getLogicalStart());
          });
        });
        if (!pairedComplete)
          return true;
      }
    }
  }
  return false;
}

ArrayAttr reductionSources(ReduceOp reduce) {
  SmallVector<Attribute> sources;
  for (Value input : reduce.getSources()) {
    auto fragment = dyn_cast<FragmentType>(input.getType());
    if (!fragment)
      continue;
    for (int64_t axis : reduce.getAxes()) {
      if (axis < 0 || axis >= static_cast<int64_t>(fragment.getAxisMaps().size()))
        continue;
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      auto source = PhysicalSourceAttr::get(
          reduce.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDerived());
      if (!llvm::is_contained(sources, Attribute(source)))
        sources.push_back(source);
    }
  }
  return ArrayAttr::get(reduce.getContext(), sources);
}

MakeRangeOp sourceRange(Value value) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return {};
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.sourceRanges(value);
  FailureOr<MakeRangeOp> range = queryExactLogicalRange(fact);
  return succeeded(range) ? *range : MakeRangeOp();
}

FailureOr<Value> scalarSource(Value value) {
  if (!value)
    return failure();
  if (!isa<FragmentType>(value.getType()))
    return value;
  while (auto extract = value.getDefiningOp<ExtractOp>()) {
    auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
    if (!record || extract.getField() >= record.getFields().size())
      return failure();
    value = record.getFields()[extract.getField()];
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return scalarSource(broadcast.getValue());
  if (auto splat = value.getDefiningOp<SplatOp>())
    return splat.getValue();
  return failure();
}

bool sameScalarValue(Value lhs, Value rhs) {
  FailureOr<Value> left = scalarSource(lhs);
  FailureOr<Value> right = scalarSource(rhs);
  if (failed(left) || failed(right))
    return false;
  if (*left == *right)
    return true;
  auto leftConstant = (*left).getDefiningOp<arith::ConstantOp>();
  auto rightConstant = (*right).getDefiningOp<arith::ConstantOp>();
  return leftConstant && rightConstant &&
         leftConstant.getValue() == rightConstant.getValue();
}

bool isReplayableWithoutLoad(Value value, PhysicalSourceAxis source) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return false;
  if (!queryFragmentAxis(value.getType(), source).isExact())
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  return analysis
      .replayability(value, source, PhysicalReplayScope::ValueGraph,
                     /*allowAccesses=*/false)
      .isReplayable();
}

namespace {

bool isUnitStep(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value() == 1;
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<RangeOp>();
    return range && bound.getBound() == 2 && isUnitStep(range.getStep());
  }
  return false;
}

} // namespace

FailureOr<std::optional<RootAccess>>
analyzeRoot(LoadOp load, ArrayRef<MakeRangeOp> reductionRanges,
            unsigned preferredAxis, bool diagnose) {
  auto fragment = dyn_cast<FragmentType>(load.getResult().getType());
  if (!fragment || reductionRanges.empty())
    return std::optional<RootAccess>();
  auto kernel = load->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  struct AxisRelation {
    unsigned axis;
    MakeRangeOp authority;
  };
  SmallVector<AxisRelation> resultAxes;
  SmallVector<AxisRelation> directResultAxes;
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    PhysicalRangeFact fact = analysis.axisRanges(load.getResult(), axis);
    if (fact.roots.empty())
      continue;
    SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
    combined.append(reductionRanges.begin(), reductionRanges.end());
    PhysicalLockstepTraversalFact relation = analysis.lockstepRanges(combined);
    if (relation.isExact()) {
      resultAxes.push_back({axis, fact.roots.front()});
      if (llvm::any_of(fact.roots, [&](MakeRangeOp range) {
            return llvm::is_contained(reductionRanges, range);
          }))
        directResultAxes.push_back({axis, fact.roots.front()});
    }
  }
  if (!directResultAxes.empty())
    resultAxes = std::move(directResultAxes);
  if (resultAxes.empty())
    return std::optional<RootAccess>();
  if (resultAxes.size() > 1) {
    llvm::erase_if(resultAxes,
                   [&](const AxisRelation &relation) {
                     return relation.axis != preferredAxis;
                   });
  }
  if (resultAxes.size() != 1) {
    if (diagnose)
      load.emitOpError("reduction traversal has no unique root-load fragment axis");
    return failure();
  }
  unsigned reductionAxis = resultAxes.front().axis;
  MakeRangeOp resultAuthority = resultAxes.front().authority;
  SmallVector<std::pair<unsigned, unsigned>, 2> occurrences;
  SmallVector<std::pair<unsigned, unsigned>, 2> directOccurrences;
  for (auto [index, value] : llvm::enumerate(load.getCoordinates())) {
    auto type = dyn_cast<FragmentType>(value.getType());
    if (!type)
      continue;
    for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
      PhysicalRangeFact fact = analysis.axisRanges(value, axis);
      if (fact.roots.empty())
        continue;
      if (llvm::is_contained(fact.roots, resultAuthority))
        directOccurrences.emplace_back(index, axis);
      SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
      combined.push_back(resultAuthority);
      if (analysis.lockstepRanges(combined).isExact())
        occurrences.emplace_back(index, axis);
    }
  }
  // Equal bounds do not identify one of two independent Cartesian axes.
  if (!directOccurrences.empty())
    occurrences = std::move(directOccurrences);
  SmallVector<std::pair<unsigned, unsigned>, 2> alignedOccurrences;
  llvm::copy_if(occurrences, std::back_inserter(alignedOccurrences),
                [&](const auto &occurrence) {
                  return occurrence.second == reductionAxis;
                });
  if (!alignedOccurrences.empty())
    occurrences = std::move(alignedOccurrences);
  bool projectsToReductionAxis = llvm::all_of(occurrences, [&](const auto &occurrence) {
    auto type = cast<FragmentType>(load.getCoordinates()[occurrence.first].getType());
    auto projection = queryAxisProjection(type, fragment);
    return projection.isExact() &&
           projection.targetToSource[reductionAxis] == occurrence.second;
  });
  if (occurrences.empty() || !projectsToReductionAxis) {
    if (!diagnose) return failure();
    InFlightDiagnostic diagnostic = load.emitOpError(
        "reduction source provenance is absent from load coordinates");
    diagnostic << "; reduction_range=" << resultAuthority.getResult().getType()
               << ", fragment_axis=" << reductionAxis
               << ", matching_occurrences=" << occurrences.size();
    for (Value value : load.getCoordinates())
      diagnostic << ", coordinate=" << value.getType();
    return failure();
  }
  if (!isUnitStep(resultAuthority.getStep())) {
    if (diagnose) load.emitOpError("reduction load range is not unit-step");
    return failure();
  }
  return std::optional<RootAccess>(RootAccess{
      load, resultAuthority, occurrences.front().first, reductionAxis});
}

FailureOr<SourcePlan> analyzeSource(Value source, unsigned reductionAxis,
                                  bool diagnose) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || reductionAxis >= fragment.getShape().size())
    return failure();
  FailureOr<AxisMapAttr> mapping = queryAxisMap(fragment, reductionAxis);
  if (failed(mapping))
    return failure();
  SourcePlan plan{source, sourceAxisIdentity(*mapping), reductionAxis, {}, {}, {}};
  auto kernel = source.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.axisRanges(source, reductionAxis);
  PhysicalLockstepTraversalFact relation = analysis.lockstepRanges(fact.roots);
  if (!relation.isExact())
    return failure();
  plan.reductionRange = relation.authority;
  for (MakeRangeOp range : fact.roots)
    if (!llvm::is_contained(plan.ranges, range))
      plan.ranges.push_back(range);
  // Predicates, gathers and address arithmetic can carry another occurrence of
  // the same traversal without defining the result fragment axis.  Replay must
  // map those coordinates from the same lockstep authority instead of meeting
  // an unmapped make_range inside the chunk loop.
  PhysicalRangeFact graphRanges = analysis.sourceRanges(source);
  llvm::SmallPtrSet<Operation *, 8> otherAxisRanges;
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    if (axis == reductionAxis)
      continue;
    PhysicalRangeFact other = analysis.axisRanges(source, axis);
    for (MakeRangeOp range : other.roots)
      if (!llvm::is_contained(plan.ranges, range))
        otherAxisRanges.insert(range.getOperation());
  }
  for (MakeRangeOp range : graphRanges.roots) {
    if (llvm::is_contained(plan.ranges, range) ||
        otherAxisRanges.contains(range.getOperation()))
      continue;
    if (!llvm::any_of(plan.ranges, [&](MakeRangeOp selected) {
          return sameLogicalRange(range, selected);
        }))
      continue;
    SmallVector<MakeRangeOp> combined(plan.ranges.begin(), plan.ranges.end());
    combined.push_back(range);
    if (analysis.lockstepRanges(combined).isExact())
      plan.ranges.push_back(range);
  }
  for (Operation *access : fact.accesses) {
    auto load = dyn_cast<LoadOp>(access);
    if (!load || llvm::is_contained(plan.roots, load))
      continue;
    FailureOr<std::optional<RootAccess>> root =
        analyzeRoot(load, plan.ranges, reductionAxis, diagnose);
    if (failed(root))
      return failure();
    if (*root)
      plan.roots.push_back(load);
  }
  if (plan.roots.empty() && plan.ranges.empty())
    return failure();
  return plan;
}

FailureOr<ParameterAttr> parameterForExtent(func::FuncOp kernel,
                                          PhysicalExprAttr extent) {
  if (extent.getKind() !=
      PhysicalExprKind::Parameter)
    return failure();
  ParameterAttr result = lookupParameter(kernel, extent.getParameterReference());
  return result ? FailureOr<ParameterAttr>(result)
                : FailureOr<ParameterAttr>(failure());
}

bool hasNonUnitFreeAxis(ReduceOp reduce) {
  for (Value source : reduce.getSources()) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment)
      continue;
    for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
      if (llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
        continue;
      auto extent = cast<PhysicalExprAttr>(attribute);
      if (extent.getKind() ==
              PhysicalExprKind::Constant &&
          extent.getValue() == 1)
        continue;
      return true;
    }
  }
  return false;
}

bool hasSelectedSegmentExtent(ReduceOp reduce, func::FuncOp kernel) {
  ParameterAttr segment;
  for (Value source : reduce.getSources()) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    int64_t axis = reduce.getAxes().empty() ? -1 : reduce.getAxes().front();
    if (!fragment || axis < 0 ||
        axis >= static_cast<int64_t>(fragment.getShape().size()))
      return false;
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    FailureOr<ParameterAttr> parameter = parameterForExtent(kernel, extent);
    if (failed(parameter) ||
        parameter->getRole() !=
            ParameterRole::ScanChunk)
      return false;
    if (segment && segment != *parameter)
      return false;
    segment = *parameter;
  }
  return static_cast<bool>(segment);
}

FailureOr<Value> dimensionArgument(func::FuncOp kernel, int64_t dimension) {
  if (auto argument = resolveDimension(kernel, dimension)) return Value(argument);
  return failure();
}

bool sameFullOrdinalTraversal(ArrayRef<MakeRangeOp> ranges) {
  if (ranges.empty())
    return false;
  auto cardinality = constantLogicalRangeCardinality(ranges.front());
  if (!cardinality)
    return false;
  MakeRangeOp first = ranges.front();
  return llvm::all_of(ranges, [&](MakeRangeOp range) {
    return constantLogicalRangeCardinality(range) == cardinality &&
           samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()) &&
           samePhysicalScalarExpression(range.getExtent(), first.getExtent());
  });
}

FailureOr<SourcePlan> nestedScalarReductionSource(ReduceOp reduce,
                                                func::FuncOp kernel) {
  if (reduce.getAxes() != ArrayRef<int64_t>{0} ||
      reduce.getSources().size() != 1 || reduce.getNumResults() != 1 ||
      isa<FragmentType, RecordType>(reduce.getResult(0).getType()))
    return failure();
  Value source = reduce.getSources().front();
  auto type = dyn_cast<FragmentType>(source.getType());
  if (!type || type.getShape().size() != 1)
    return failure();
  FailureOr<ParameterAttr> parameter = parameterForExtent(
      kernel, cast<PhysicalExprAttr>(type.getShape()[0]));
  PhysicalProgramAnalysis analysis(kernel);
  if (succeeded(parameter)) {
    if (!parameter->isDeferred())
      return failure();
  }
  FailureOr<SourcePlan> plan = analyzeSource(source, 0);
  if (failed(plan) ||
      !llvm::all_of(plan->ranges, [](MakeRangeOp range) {
        return isUnitStepRange(range);
      }))
    return failure();
  PhysicalReplayFact replay = analysis.replayAt(
      source, plan->sourceIdentity, PhysicalReplayScope::ValueGraph,
      /*allowAccesses=*/true, reduce, IRMapping{});
  if (!replay.isReplayable() ||
      llvm::any_of(replay.accesses, [&](Operation *access) {
        return !isa<LoadOp>(access);
      }))
    return failure();

  SmallVector<Value> values{source};
  llvm::SmallPtrSet<Operation *, 32> producers;
  bool hasNestedReduction = false;
  for (unsigned index = 0; index < values.size(); ++index) {
    Value value = values[index];
    Operation *producer = value.getDefiningOp();
    if (!producer || !producers.insert(producer).second)
      continue;
    if (isa<ScanOp>(producer))
      return failure();
    if (auto nested = dyn_cast<ReduceOp>(producer))
      for (Value result : nested.getResults()) {
        auto fragment = dyn_cast<FragmentType>(result.getType());
        if (fragment &&
            !queryFragmentAxes(fragment, plan->sourceIdentity).empty())
          hasNestedReduction = true;
      }
    values.append(producer->getOperands().begin(), producer->getOperands().end());
  }
  if (!hasNestedReduction)
    return failure();
  if (failed(parameter) &&
      !analysis.axisRealization(source, 0).constructionScalarSeed &&
      !llvm::any_of(values, [&](Value value) {
        return exceedsRegisterFile(value, kernel);
      }))
    return failure();
  // Shared one-dimensional ancestors may remain outside the traversal. Nested
  // reductions and their multidimensional producers must move as a closed slice.
  SmallVector<Operation *> retained;
  llvm::SmallPtrSet<Operation *, 32> visited;
  for (Operation *producer : producers)
    for (Operation *user : producer->getUsers())
      if (user != reduce && !producers.contains(user) &&
          visited.insert(producer).second)
        retained.push_back(producer);
  for (unsigned index = 0; index < retained.size(); ++index) {
    Operation *producer = retained[index];
    if (isa<ReduceOp, ScanOp, ContractOp, ScaledContractOp, SparseContractOp>(
            producer))
      return failure();
    for (Value result : producer->getResults()) {
      if (isa<RecordType>(result.getType()))
        return failure();
      auto fragment = dyn_cast<FragmentType>(result.getType());
      if (!fragment)
        continue;
      unsigned varyingAxes = 0;
      for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
        auto extent = cast<PhysicalExprAttr>(attribute);
        if (extent.getKind() !=
                PhysicalExprKind::Constant ||
            extent.getValue() != 1 ||
            analysis.axisRealization(result, axis).constructionScalarSeed)
          ++varyingAxes;
      }
      if (varyingAxes > 1)
        return failure();
    }
    for (Value operand : producer->getOperands())
      if (Operation *ancestor = operand.getDefiningOp();
          ancestor && visited.insert(ancestor).second)
        retained.push_back(ancestor);
  }
  return plan;
}

namespace {

bool sameCombineType(Type lhs, Type rhs) {
  lhs = uniformElementType(lhs);
  rhs = uniformElementType(rhs);
  if (auto record = dyn_cast<RecordType>(lhs)) {
    auto other = dyn_cast<RecordType>(rhs);
    if (!other || record.getFieldNames() != other.getFieldNames()) return false;
    return llvm::all_of(llvm::zip(record.getFieldTypes(), other.getFieldTypes()),
        [](auto pair) {
          return sameCombineType(cast<TypeAttr>(std::get<0>(pair)).getValue(),
                                 cast<TypeAttr>(std::get<1>(pair)).getValue());
        });
  }
  return lhs == rhs;
}

} // namespace

bool matchReductionCombine(ReduceOp reference, Block *body, ValueRange left,
    SmallVectorImpl<Value> &right, ValueRange results,
    llvm::SmallPtrSetImpl<Operation *> &matched) {
  unsigned count = reference.getSources().size();
  Block &combine = reference.getCombine().front();
  if (!reference.getCaptures().empty() || left.size() != count ||
      right.size() != count || results.size() != count ||
      combine.getNumArguments() != count * 2) return false;
  IRMapping mapping;
  auto structured = cast<StructuredOpInterface>(reference.getOperation());
  auto rhs = structured.getCombineRhs();
  for (unsigned i = 0; i < count; ++i) {
    mapping.map(structured.getCombineLhs()[i], left[i]);
    if (right[i]) mapping.map(rhs[i], right[i]);
  }
  SmallVector<Value> inferred(right.begin(), right.end());
  llvm::SmallPtrSet<Operation *, 32> updates;
  std::function<bool(Value, Value)> match = [&](Value pattern, Value value) {
    if (!sameCombineType(pattern.getType(), value.getType())) return false;
    if (Value mapped = mapping.lookupOrNull(pattern)) return mapped == value;
    if (isa<BlockArgument>(pattern)) {
      auto found = llvm::find(rhs, pattern);
      if (found == rhs.end()) return false;
      inferred[std::distance(rhs.begin(), found)] = value;
      mapping.map(pattern, value);
      return true;
    }
    auto scalar = scalarSource(pattern);
    if (succeeded(scalar) && (*scalar).getDefiningOp<arith::ConstantOp>())
      return sameScalarValue(pattern, value);
    Operation *expected = pattern.getDefiningOp(), *actual = value.getDefiningOp();
    if (!expected || !actual || actual->getBlock() != body ||
        !isLaneWiseValueOperation(expected) ||
        expected->getName() != actual->getName() ||
        expected->getNumOperands() != actual->getNumOperands() ||
        expected->getNumResults() != actual->getNumResults()) return false;
    NamedAttrList expectedAttrs(expected->getAttrs()), actualAttrs(actual->getAttrs());
    expectedAttrs.erase(originAttr);
    actualAttrs.erase(originAttr);
    if (expectedAttrs != actualAttrs) return false;
    for (auto [a, b] : llvm::zip(expected->getOperands(), actual->getOperands()))
      if (!match(a, b)) return false;
    for (auto [a, b] : llvm::zip(expected->getResults(), actual->getResults())) {
      if (!sameCombineType(a.getType(), b.getType())) return false;
      mapping.map(a, b);
    }
    updates.insert(actual);
    return mapping.lookup(pattern) == value;
  };
  for (auto [expected, actual] : llvm::zip(
           cast<YieldOp>(combine.getTerminator()).getValues(), results))
    if (!match(expected, actual)) return false;
  if (!llvm::all_of(inferred, [](Value value) { return bool(value); })) return false;
  right.assign(inferred.begin(), inferred.end());
  matched.insert(updates.begin(), updates.end());
  return true;
}

} // namespace intent::gpu::reduction
